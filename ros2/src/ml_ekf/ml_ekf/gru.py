import json
import time
import torch
import argparse
import numpy as np

from torch import nn
from pathlib import Path
from .ekf import EKF, AugServoArmModel, ServoArmModel, run_ekf

Z95 = 1.959963984540054
FEATURES = ("sin_theta", "cos_theta", "omega", "d", "r_minus_theta", "r_change", "last_innov", "steps_since_meas", "log_sd_omega", "log_sd_d")

# ************************************** #
# **************** Data **************** #
# ************************************** #
def load(path, n_sub: int = 10) -> dict:
    """A recorded run, with r_sub[k, j] = the target commanded at t_k + j dt / n_sub (from the raw messages)."""
    d = np.load(path)
    out = {}

    # copy the four arrays we need out of the .npz file
    for key in ("t", "z", "theta_true", "omega_true"):
        out[key] = d[key] # t: timestamps, z: sensor readings (NaN = none), true angle and speed

    out["dt"] = float(d["dt"]) # step size (0.01 s); stored as a 0-d array in the .npz, so make it a plain float
    out["name"] = Path(path).stem # file name without folder or .npz, e.g. "steps_2000", used in the results table

    tg = d["raw_target"]
    ts = out["t"][:, None] + out["dt"] * np.arange(n_sub) / n_sub
    i = np.searchsorted(tg[:, 0], ts, side="right") - 1

    out["r_sub"] = np.where(i >= 0, tg[np.clip(i, 0, None), 1], tg[0, 1])
    return out


def label(d: dict, nominal: ServoArmModel) -> np.ndarray:
    # This measures and labels the real missing acceleration (the GRU's target), this is meant only for training targets, not for live use.
    T = len(d["t"])
    dt = d["dt"]
    a = np.full(T, np.nan)

    for k in range(T - 1):
        x, _ = nominal.step(np.array([d["theta_true"][k], d["omega_true"][k]]), d["r_sub"][k], dt)
        a[k] = (d["omega_true"][k + 1] - x[1]) / dt
    return a

def feature_row(x, P, r_sub_k, r_prev, innov, since) -> np.ndarray:
    """What the filter knows at t_k, as one feature vector (order = FEATURES).

    x --> filter state [theta, omega, d] at t_k
    P --> filter covariance (3x3) at t_k
    r_sub_k --> servo targets commanded over this step (n_sub values); [-1] = the latest
    r_prev --> latest target of the previous step
    innov --> last normalised innovation (z - theta_pred) / sqrt(S), held until the next measurement
    since --> filter steps since the last measurement (0 right after one)
    """
    return np.array([
        np.sin(x[0]), # sin(theta): gravity torque is mgl sin(theta); sin/cos avoid angle wrap jumps
        np.cos(x[0]), # cos(theta): with sin, pins down the angle smoothly
        x[1], # omega [rad/s]: torque limit and friction depend on speed
        x[2], # d [rad/s^2]: B1-aug's own running guess of the missing acceleration
        r_sub_k[-1] - x[0], # tracking error r - theta [rad]: what the servo reacts to; large = torque saturates
        r_sub_k[-1] - r_prev, # command change this step [rad]: flags a new step, when delay / lag matter most
        innov, # last measurement surprise (in std devs): recent model error and its sign
        np.log1p(since), # log(1 + steps since last reading): how stale the estimate is
        0.5 * np.log(P[1, 1]), # log std of omega: how unsure the filter is about speed
        0.5 * np.log(P[2, 2]), # log std of d: how unsure the filter is about the missing push
    ], np.float32)

def features_from_b1aug(d: dict, q, R: float) -> np.ndarray:
    # (T, F) features from a plain B1-aug run over the recording (the GRU's training inputs).
    z0 = d["z"][np.isfinite(d["z"])][0] # First sensor measurement, to innit the filter state. The GRU should not see the true state, onle the filter's own estimate.

    res = run_ekf(AugServoArmModel(), d["z"], d["r_sub"], d["dt"], q, R, x0=[z0, 0.0, 0.0])

    T = len(d["t"])
    X = np.zeros((T, len(FEATURES)), np.float32)
    innov, since = 0.0, 0
    for k in range(T):
        if np.isfinite(res.innov[k]):
            innov = res.innov[k] / np.sqrt(res.S[k]) # textbook Kalman filter innovation and innovation covariance
            since = 0
        else:
            since += 1

        if k:
            r_prev = d["r_sub"][k - 1][-1]
        else:
            r_prev = d["r_sub"][0][0]

        X[k] = feature_row(res.x[k], res.P[k], d["r_sub"][k], r_prev, innov, since)
    return X

# ******************************************** #
#  **************** NetworGRU **************** #
# ******************************************** #
class GRUCorrector(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)
        self.hidden = hidden

    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        output = self.out(y)
        return output[..., 0], output[..., 1].clamp(-8, 6), h

def train_gru(X_train, y_train, X_val, y_val, epochs: int, chunk: int, hidden: int, lr: float, device, seed: int, log=print):
    torch.manual_seed(seed)

    net = GRUCorrector(X_train[0].shape[1], hidden).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr) # Optimzer Adam, learning rate = lr

    def chunks(Xs, ys): # cut every run into chunk-long pieces (NaN labels masked)
        xs, yy = [], []
        for X, y in zip(Xs, ys):
            for i in range(0, len(X) - chunk + 1, chunk // 2):
                xs.append(X[i:i + chunk])
                yy.append(y[i:i + chunk])

        return torch.tensor(np.stack(xs), device=device), torch.tensor(np.stack(yy), device=device)

    def nll(net, X, y):
        mu, lv, _ = net(X)
        m = torch.isfinite(y)
        return (0.5 * ((torch.nan_to_num(y) - mu) ** 2 * torch.exp(-lv) + lv))[m].mean()

    Xc, yc = chunks(X_train, y_train)
    Xv, yv = chunks(X_val, y_val)

    best = float("inf")
    best_state = None

    g = torch.Generator(device="cpu").manual_seed(seed) # Random number generator

    for ep in range(1, epochs + 1):
        net.train() #nn Model train() makes it ready for training

        perm = torch.randperm(len(Xc), generator=g) # Shuffle training chunk indices each epoch
        losses = []

        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = nll(net, Xc[b], yc[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        net.eval()

        with torch.no_grad():
            v = nll(net, Xv, yv).item()

        if v < best:
            best = v
            best_state = {}

            for k, t in net.state_dict().items():
                best_state[k] = t.detach().clone()

        log(f"  epoch {ep:3d}  train NLL {np.mean(losses):.3f}  val NLL {v:.3f}  (best {best:.3f})")

    net.load_state_dict(best_state)

    return net.cpu().eval()

# ********************************************** #
#  **************** EKF Filters **************** #
# ********************************************** #
class B3Model(AugServoArmModel):
    a_extra: float = 0.0

    def f(self, x, r):
        return super().f(x, r) + np.array([0.0, self.a_extra, 0.0])

def run_b3(d: dict, net: GRUCorrector, norm: dict, q, R: float) -> tuple[np.ndarray, np.ndarray]:
    # B3 over one recording: each step, GRU(features at k-1) -> a_hat, then predict k-1 -> k and update.
    model = B3Model()
    z0 = d["z"][np.isfinite(d["z"])][0] # Initial measurement

    f = EKF(model, d["dt"], q, R, x0=[z0, 0.0, 0.0])

    T  = len(d["t"])
    dt = d["dt"]

    G = np.array([0.5 * dt * dt, dt, 0.0])

    mu_x = norm["mu"]
    sd_x = norm["sd"]
    s_y = norm["y_scale"]

    xs = np.zeros((T, 3))
    Ps = np.zeros((T, 3, 3))

    h = None
    innov = 0.0
    since = 0

    if np.isfinite(d["z"][0]):
        f.update(d["z"][0])

    xs[0] = f.x
    Ps[0] = f.P

    with torch.no_grad():
        for k in range(1, T):
            if k > 1:
                r_prev = d["r_sub"][k - 2][-1]
            else:
                r_prev = d["r_sub"][0][0]

            phi = feature_row(f.x, f.P, d["r_sub"][k - 1], r_prev, innov, since)

            mu, lv, h = net(torch.from_numpy((phi - mu_x) / sd_x)[None, None], h)

            model.a_extra = mu.item() * s_y
            f.predict(d["r_sub"][k - 1])
            f.P = f.P + float(torch.exp(lv).item()) * s_y**2 * np.outer(G, G) # the GRU's own uncertainty

            if np.isfinite(d["z"][k]):
                y, S = f.update(d["z"][k])
                innov = y / np.sqrt(S)
                since = 0
            else:
                since += 1
            xs[k] = f.x
            Ps[k] = f.P
    return xs, Ps


def run_b1aug(d: dict, q, R: float):
    z0 = d["z"][np.isfinite(d["z"])][0]
    res = run_ekf(AugServoArmModel(), d["z"], d["r_sub"], d["dt"], q, R, x0=[z0, 0.0, 0.0])
    return res.x, res.P


def score(x, P, d: dict, skip: float) -> dict[str, float]:
    k0 = int(round(skip / d["dt"]))
    x = x[k0:, :2]
    P = P[k0:, :2, :2]

    e = np.stack([x[:, 0] - d["theta_true"][k0:], x[:, 1] - d["omega_true"][k0:]], 1)
    sd = np.sqrt(np.stack([P[:, 0, 0], P[:, 1, 1]], 1))
    nees = np.einsum("ki,kij,kj->k", e, np.linalg.inv(P), e)
    nll = 0.5 * (nees + np.log(np.linalg.det(2 * np.pi * P)))
    return {"nll": float(nll.mean()), "rmse_theta_deg": float(np.degrees(np.sqrt(np.mean(e[:, 0] ** 2)))),
            "rmse_omega": float(np.sqrt(np.mean(e[:, 1] ** 2))),
            "cov95_theta": float(np.mean(np.abs(e[:, 0]) <= Z95 * sd[:, 0])),
            "cov95_omega": float(np.mean(np.abs(e[:, 1]) <= Z95 * sd[:, 1])), "nees": float(nees.mean())}

# *************** #
# ****  main **** #
# *************** #
def main() -> None:
    ap = argparse.ArgumentParser(description="Train and score B3 (GRU dynamics correction on B1-aug).")
    ap.add_argument("--data", default="~/ml_ekf_runs/dataset", help="folder with train/ val/ test/ ood/")
    ap.add_argument("--out", default="~/ml_ekf_runs/gru_dyn")
    ap.add_argument("--q-omega", type=float, default=10.0, help="B1-aug q_omega (tuned with eval_ekf)")
    ap.add_argument("--q-d", type=float, default=3.16e5, help="B1-aug q_d (tuned with eval_ekf)")
    ap.add_argument("--sigma", type=float, default=0.01, help="sensor noise std [rad]")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--chunk", type=int, default=200, help="steps per training sequence (2 s)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=float, default=1.0, help="seconds left out of the scores at the start")
    ap.add_argument("--no-retune", action="store_true", help="keep q_d for B3 instead of re-tuning it on val")
    a = ap.parse_args()

    data = Path(a.data).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print(f"[b3] using GPU {torch.cuda.get_device_name(0)}", flush=True)
        device = "cuda"
    else:
        print("[b3] using CPU", flush=True)
        device = "cpu"

    # Filter settings (q process noise)
    # q_omega: process noise on omega [(rad/s)^2/s], random unmodelled torque
    # q_d process noise on d [(rad/s^2)^2/s], how fast the missing-push estimate may change
    q_aug = [0.0, a.q_omega, a.q_d]

    R = a.sigma**2 # measurement noise variance (its the sensor's noise squared, not the filter's innovation variance)

    nominal = ServoArmModel()

    # Dataset Splitting ( We already have train/val/test/ood folders from the simulatiion, so just load them all )
    splits = {}

    # OOD means out-of-distribution: data that differs from the conditions used during training.
    for split_name in ("train", "val", "test", "ood"):
        split_directory = data.joinpath(split_name)
        file_paths = sorted(split_directory.glob("*.npz"))

        loaded_data = []
        for file_path in file_paths:
            loaded_data.append(load(file_path))

        splits[split_name] = loaded_data

    run_counts = []

    for split_name, loaded_data in splits.items():
        count = len(loaded_data)
        split_summary = f"{split_name} {count}"
        run_counts.append(split_summary)

    runs_summary = ", ".join(run_counts)
    message = f"[b3] runs: {runs_summary}; device {device}"

    print(message, flush=True)


    # Begin Time
    t0 = time.time()

    feats = {}

    for split_name in ("train", "val"):
        split_features = []

        for run_data in splits[split_name]:
            run_features = features_from_b1aug(run_data, q_aug, R)
            split_features.append(run_features)

        feats[split_name] = split_features

    labels = {}

    for split_name in ("train", "val"):
        split_labels = []

        for run_data in splits[split_name]:
            run_labels = label(run_data, nominal)
            split_labels.append(run_labels)

        labels[split_name] = split_labels

    # Normalisation, from the TRAIN split only (val/test/ood must not leak in); saved in gru.pt for live use
    allX = np.concatenate(feats["train"])  # every train time step in one table: (steps, 10 features)

    mu = allX.mean(0) # mean of each feature, the GRU sees (x - mu) / sd, roughly in [-1, 1]
    sd = allX.std(0) + 1e-6  # spread of each feature, + 1e-6 avoids dividing by zero for a constant feature
    y_scale = float(np.nanstd(np.concatenate(labels["train"])))  # spread of the missing acceleration [rad/s^2], GRU learns a / y_scale (nanstd: each run's last label is NaN)

    print(f"[b3] features and labels ready ({time.time() - t0:.0f} s); missing acceleration std {y_scale:.1f} rad/s^2", flush=True)

    def normalize(Xs):
        normalized_features = []

        for X in Xs:
            normalized = (X - mu) / sd
            normalized = normalized.astype(np.float32)
            normalized_features.append(normalized)

        return normalized_features

    def scale(ys):
        scaled_labels = []

        for y in ys:
            scaled = y / y_scale
            scaled = scaled.astype(np.float32)
            scaled_labels.append(scaled)

        return scaled_labels

    t0 = time.time()

    # Train the GRU
    net = train_gru(normalize(feats["train"]),
                    scale(labels["train"]),
                    normalize(feats["val"]),
                    scale(labels["val"]),
                    a.epochs, a.chunk,
                    a.hidden, a.lr, device, a.seed)

    print(f"[b3] trained {sum(p.numel() for p in net.parameters())} parameters in {time.time() - t0:.0f} s", flush=True)
    norm = {"mu": mu, "sd": sd, "y_scale": y_scale}

    # Hyperparameter retuning for B3: q_d is the only one that matters, and it is tuned on the val split (the other two are fixed to the B1-aug values).
    q_b3 = q_aug
    if not a.no_retune:
        grid = []
        for f in (1e-3, 1e-2, 1e-1, 1.0):
            grid.append(a.q_d * f)

        val_nll = []  # same criterion as eval_ekf: accurate AND honest about the uncertainty

        for qd in grid:
            ss = []
            for d in splits["val"]:
                result = run_b3(d, net, norm, [0.0, a.q_omega, qd], R)
                s = score(*result, d, a.skip)
                ss.append(s)

            nll_values = []
            for x in ss:
                nll_values.append(x["nll"])

            val_nll.append(np.mean(nll_values))

            print(f"  val q_d {qd:.3g}: NLL {val_nll[-1]:.3f}, omega RMSE {np.mean([x['rmse_omega'] for x in ss]):.3f}", flush=True)

        q_b3 = [0.0, a.q_omega, grid[int(np.argmin(val_nll))]]

    print(f"[b3] B3 uses q_omega {q_b3[1]:.3g}, q_d {q_b3[2]:.3g}", flush=True)

    # Score the filters on the test and OOD splits
    results = {}

    hdr = f"{'split':6s} {'filter':7s} {'theta [deg]':>11s} {'omega [rad/s]':>13s} {'cov95 th':>9s} {'cov95 om':>9s} {'NEES':>6s}"
    print("\n" + hdr)

    for s in ("test", "ood"):
        for name, fn in (("B1-aug", lambda d: run_b1aug(d, q_aug, R)), ("B3", lambda d: run_b3(d, net, norm, q_b3, R))):
            ss = []
            for d in splits[s]:
                ss.append(score(*fn(d), d, a.skip))

            m = {}
            for k in ss[0]:
                values = []
                for x in ss:
                    values.append(x[k])
                m[k] = float(np.mean(values))

            results.setdefault(s, {})[name] = {"mean": m, "runs": {d["name"]: x for d, x in zip(splits[s], ss)}}

            print(f"{s:6s} {name:7s} {m['rmse_theta_deg']:11.3f} {m['rmse_omega']:13.3f} {m['cov95_theta']:9.2f} "
                  f"{m['cov95_omega']:9.2f} {m['nees']:6.2f}", flush=True)

        b = results[s]["B1-aug"]["mean"]
        g = results[s]["B3"]["mean"]

        print(f"{s:6s} change  {100 * (g['rmse_theta_deg'] / b['rmse_theta_deg'] - 1):+10.0f}% "
              f"{100 * (g['rmse_omega'] / b['rmse_omega'] - 1):+12.0f}%")

    torch.save({"state_dict": net.state_dict(), "features": FEATURES, "hidden": a.hidden, "mu": mu, "sd": sd, "y_scale": y_scale, "q_b1aug": q_aug, "q_b3": q_b3, "sigma": a.sigma}, out / "gru.pt")

    (out / "results.json").write_text(json.dumps({"args": vars(a), "q_b3": q_b3, "results": results}, indent=1))

    print(f"\n[b3] saved {out / 'gru.pt'} and {out / 'results.json'}")

if __name__ == "__main__":
    main()
