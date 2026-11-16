# Dataset: record many seeded Gazebo runs, one .npz each, split into train / val / test / ood.
# ros2 run ml_ekf make_dataset --out ~/ml_ekf_runs/dataset # all splits
# ros2 run ml_ekf make_dataset --out ~/ml_ekf_runs/dataset --splits val --dry-run # print the commands only

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

MODES = ("steps", "sine", "chirp")
SPLITS = {"train": 0, "val": 1000, "test": 2000, "ood": 3000} # first seed of each split
OOD_SERVO = {"delay": 0.025, "tau_stall": 0.75, "deadband": 0.02} # true_servo defaults: 0.01, 1.0, 0.01

def command_params(mode: str, seed: int) -> dict:
    # Random shape of one run's command; mid 1.55 rad, so mid +- amp stays inside the 0.3..2.8 rad range.
    rng = np.random.default_rng([seed, MODES.index(mode)])
    if mode == "steps":
        return {"hold": round(float(rng.uniform(0.4, 1.5)), 3)}
    if mode == "sine":
        return {"amp": round(float(rng.uniform(0.3, 1.2)), 3), "freq": round(float(rng.uniform(0.2, 2.0)), 3)}
    return {"amp": round(float(rng.uniform(0.3, 1.0)), 3), "f0": round(float(rng.uniform(0.1, 0.5)), 3),
            "f1": round(float(rng.uniform(1.5, 4.0)), 3), "sweep": round(float(rng.uniform(10.0, 30.0)), 2)}

def kv(d: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in d.items())

def launch_cmd(mode, seed, path, duration, rtf, cmd_p, servo_p, estimators) -> list[str]:
    cmd = ["ros2", "launch", "ml_ekf", "display.launch.py", "gazebo:=true", "headless:=true",
           f"command:={mode}", f"seed:={seed}", f"record:={path}", f"duration:={duration}", f"rtf:={rtf}",
           f"command_params:={kv(cmd_p)}"]
    if servo_p:
        cmd.append(f"servo_params:={kv(servo_p)}")
    if estimators:
        cmd.append(f"estimators:={estimators}")
    return cmd

def run(cmd: list[str], timeout: float, log: Path) -> bool:
    # One launch in its own process group, so a hung Gazebo can be killed with everything it started.
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout) == 0
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            return False

def check(path: Path, duration: float) -> str | None:
    # None if the file looks like a full run, else what is wrong with it.
    if not path.exists():
        return "no file"
    d = np.load(path)
    span = d["t"][-1] - d["t"][0]
    n_z = int(np.isfinite(d["z"]).sum())
    if span < 0.9 * duration:
        return f"only {span:.1f} s recorded"
    if n_z < 5 * span: # the sensor gives ~19 per s at 20 Hz with 5 % drops
        return f"only {n_z} measurements"
    return None

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="~/ml_ekf_runs/dataset", help="folder; one sub-folder per split")
    ap.add_argument("--splits", default="train,val,test,ood")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--n", default="train=20,val=4,test=6,ood=6", help="runs per mode in each split")
    ap.add_argument("--duration", type=float, default=30.0, help="sim seconds per run")
    ap.add_argument("--rtf", type=float, default=1.0, help="Gazebo real-time factor target")
    ap.add_argument("--estimators", default="", help="also log live filters, e.g. b1,b1aug")
    ap.add_argument("--dry-run", action="store_true", help="print the launch commands, run nothing")
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    n = {k: int(v) for k, v in (s.split("=") for s in a.n.split(","))}
    modes = [m for m in a.modes.split(",") if m]
    jobs = [(sp, m, SPLITS[sp] + i) for sp in a.splits.split(",") for m in modes for i in range(n.get(sp, 0))]
    timeout = a.duration / a.rtf * 3 + 120 # sim may run slower than rtf; plus Gazebo start-up

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    print(f"[dataset] {len(jobs)} runs of {a.duration} s into {out}", flush=True)
    t_all = time.time()
    for i, (split, mode, seed) in enumerate(jobs, 1):
        path = out / split / f"{mode}_{seed:04d}.npz"
        cmd_p = command_params(mode, seed)
        servo_p = OOD_SERVO if split == "ood" else {}
        cmd = launch_cmd(mode, seed, path, a.duration, a.rtf, cmd_p, servo_p, a.estimators)
        if a.dry_run:
            print(" ".join(cmd))
            continue
        if path.exists() and check(path, a.duration) is None:
            print(f"[{i}/{len(jobs)}] {path.name} ({split}) exists, skipped", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        ok = run(cmd, timeout, path.with_suffix(".log"))
        problem = check(path, a.duration)
        status = "ok" if problem is None else f"FAILED: {problem}" + ("" if ok else " (launch error or timeout)")
        print(f"[{i}/{len(jobs)}] {split}/{path.name}: {status} ({time.time() - t0:.0f} s)", flush=True)
        manifest[f"{split}/{path.name}"] = {"split": split, "mode": mode, "seed": seed, "duration": a.duration,
                                            "command_params": cmd_p, "servo_params": servo_p, "ok": problem is None}
        manifest_path.write_text(json.dumps(manifest, indent=1))
    if not a.dry_run:
        bad = [k for k, v in manifest.items() if not v["ok"]]
        print(f"[dataset] done in {(time.time() - t_all) / 60:.0f} min; {len(bad)} failed" + (f": {bad}" if bad else ""))

if __name__ == "__main__":
    main()
