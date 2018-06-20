#!/usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path
from platform import uname

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as file_handle:
    long_description = file_handle.read()

# common dependencies
install_req = [
    "tensorflow",
    "opencv-python",
    "ruamel.yaml",
    "matplotlib",
    "numpy"
]

setup(
    name="img2latex",
    version="0.1.0",
    description="A library to convert images into latex code",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="TODO",
    author="Bruno Schilling, Rudi Schneider",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - BETA",
        "Intended Audience 2:: Developers",
        "Topic :: Utilities",
        "License :: GPL v3",
        "Programming Language :: Python :: 3.6"
    ],
    install_requires=install_req,
    # entry_points={"console_scripts": [
    #     "nrai-train = neuroracer_ai.cli.train:main",
    #     "nrai-convert = neuroracer_ai.cli.convert:main",
    #     "nrai-augment = neuroracer_ai.cli.augment:main",
    #     "nrai-debug = neuroracer_ai.cli.debug:main"
    # ]}
)
