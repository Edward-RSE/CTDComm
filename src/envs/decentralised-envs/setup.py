from setuptools import setup, find_packages

setup(
      name='decentralised_envs',
      version='1.0.1',
      author='Jennifer Barnes-Nunn',
      author_email='jenni.barnes.nunn@gmail.com',
      description=None, #TODO: Add description and long description
      long_description=None,
      packages=find_packages(),
      install_requires=['gymnasium','pettingzoo','numpy'],  #TODO: Add specific package versions, add them to requirements.txt at the same time
)
