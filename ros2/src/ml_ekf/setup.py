from glob import glob
from setuptools import find_packages, setup

package_name = 'ml_ekf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/textures", glob("textures/*")),
        ("share/" + package_name + "/description", glob("description/*")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joaquin Philco',
    maintainer_email='joaquinphilco@hotmail.com',
    description='servo + ArUco camera rig in Gazebo Fortress',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'true_servo = ml_ekf.true_servo:main',
            'nominal_servo = ml_ekf.nominal_servo:main',
            'servo_command = ml_ekf.servo_command:main',
        ],
    },
)
