#  ___________________________________________________________________________
#
# Hybrid-RL-MPC4CLR: Hybrid Reinforcement Learning-Model Predictive Control 
#                    for Reserve Policy-Assisted Critical Load Restoration 
#                    in Distribution Grids 
# Copyright 2022 National Renewable Energy Laboratory (NREL) 
# TODO: Add copyright details including contract #, ...
#  
# This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________


import glob
import sys
import os
from setuptools import find_packages
from distutils.core import setup

# Raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

setup(
    name='rl-mpc-envs',
    version='0.0.1',
    description="Hybrid RL-MPC Environment for Learning Power System Reserve Policy to Maximize Load Restoration After Extreme Event.",
    url='https://github.nrel.gov/AGM-CSSO/rl_mpc_reserve_policy/tree/Hybrid-RL-MPC4CLR', # TODO: Update this url when the repo moves to github.com
    author='Abinet Tesfaye Eseye, Xiangyu Zhang, Bernard Knueven, Matthew Reynolds, Weijia Liu, and Wesley Jones',
    maintainer_email='Wesley.Jones@nrel.gov', # TODO: update the email if needed
    license='Revised BSD',
    packages=find_packages(),  
    scripts=[],
    include_package_data=True,
    install_requires=['tensorflow==2.2.0', 'pyomo>=6.1.2', 'tomli<2.0.0,>=0.2.6',
                      'gym', 'matplotlib', 'pandas', 'numpy', 'jupyter', 'notebook', 'pytest'],    
    python_requires='>=3.8'
    )