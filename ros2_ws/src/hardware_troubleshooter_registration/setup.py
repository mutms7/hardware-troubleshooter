from glob import glob

from setuptools import find_packages, setup

PACKAGE = "hardware_troubleshooter_registration"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Simon",
    maintainer_email="sjin2986@gmail.com",
    description="Board-edge camera registration for the hardware troubleshooter",
    license="UNLICENSED",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"registration_node = {PACKAGE}.registration_node:main",
            f"image_folder_player = {PACKAGE}.image_folder_player:main",
            f"make_synthetic_dataset = {PACKAGE}.make_synthetic_dataset:main",
        ],
    },
)
