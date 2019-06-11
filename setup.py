from setuptools import setup, find_packages

version = "1.0"

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

setup(
    name='ncpsort', 
    version='1.0', 
    packages=find_packages(exclude=[]),
    install_requires=install_requires,
    python_requires=">=3.6"
)