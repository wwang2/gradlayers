
import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="gradlayers",
    version="0.0.1",
    author="Wujie Wang",
    email="{wwj}@mit.edu",
    url="https://github.com/wwang2/gradlayers",
    packages=find_packages("."),
    scripts=[
    ],
    python_requires=">=3.5",
    install_requires=[
        "pytorch>=1.8.0",
        "numpy",
    ],
    license="MIT",
    description="Faster gradient/sensitivities without .backward()",
    long_description=read("README.md"),
)
