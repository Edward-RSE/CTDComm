from setuptools import setup, find_packages

with open("python_envs/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ctdcomm",
    version="1.0",
    packages=find_packages(where="src", include=["ctdcomm*"]),
    package_dir={"": "src"},
    install_requires=requirements,
)
