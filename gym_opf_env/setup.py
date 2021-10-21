# This software is distributed under the 3-clause BSD License.
# !/bin/usr/env python3
import glob
import sys
import os

# Raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

from setuptools import find_packages
from distutils.core import setup

packages = find_packages()

setup(
    name='opf-envs',
    version='0.0.1',
    description="RL Environments for Learning Power System Reserve Policies.",
    url='https://github.nrel.gov/AGM-CSSO', # Update this url
    author='Abinet Eseye',
    author_email='aeseye@nrel.gov',
    packages=packages,
    install_requires=['gym'], # And any other dependencies required
    )