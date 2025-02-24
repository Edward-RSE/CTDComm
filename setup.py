from setuptools import setup, find_packages

with open("python_envs/requirements.txt") as f:
    requirements = f.read().splitlines()

# Include `ctdcomm` and all subpackages in `learning_envs`
setup(
    name="CTDComm",
    version="1.0",
    packages=find_packages(include=["ctdcomm"]),
    install_requires=requirements,
)
