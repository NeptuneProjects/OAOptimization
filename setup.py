#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="oao",
    version="0.1.0",
    description="Optimization library for ocean acoustics.",
    package_dir={"": "oao"},
    packages=find_packages(where="oao"),
    url="https://github.com/NeptuneProjects/OAOptimization",
    author="William Jenkins",
    author_email="wjenkins@ucsd.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["ax-platform"],
    python_requires=">=3.10",
)
