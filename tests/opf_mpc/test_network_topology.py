"""
Distribution network topology tester.
"""

import pytest
import os
import numpy as np
import warnings

from opf_mpc_envs.envs.lr_opf_model import LROPFModel 

current_dir = os.getcwd()
os.chdir(os.path.dirname(current_dir))

lr_opf_model = LROPFModel()

class TestGridTopology:

    def test_grid_topology(self):

        bus = lr_opf_model.Bus

        r = 0
        root_bus = None

        for b in bus.keys():
            
            l = len(bus[b]['ancestor'])
            
            if l > 1:
                warnings.warn('Network Not Radial; Bus ' f"{bus[b]['index']}")
            elif l == 0:                
                root_bus = b
                r += 1   
                print("The root/substation bus is:", root_bus)                            
                
        if r == 0:
            warnings.warn("No root detected")
            root_bus = None
        elif r > 1:
            warnings.warn("More than one root detected - the network is not radial!")

        assert r == 1


# pytest test_network_topology.py