from setuptools import setup, find_packages

setup(
    name="starcraft2_envs",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["gymnasium", "pysc2"],
    license="Apache-2.0",
)
