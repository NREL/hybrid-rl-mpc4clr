"""
Distribution grid bus tester.
"""

import pytest
import os
import numpy as np
import math

from opf_mpc_envs.envs.lr_opf_model import LROPFModel 

current_dir = os.getcwd()
os.chdir(os.path.dirname(current_dir))

lr_opf_model = LROPFModel()

abs_tol = 1e-05

class TestBus:

    def test_num_bus(self):

        bus = lr_opf_model.Bus

        assert len(bus) == 13

    def test_slack_bus(self):

        bus = lr_opf_model.Bus

        assert bus[0]["is_root"] == True

    def test_simple_bus(self):

        bus = lr_opf_model.Bus

        spec = np.array([11, False, 0.018722263313609468, 0.008147651627218935, 
                         1.1025, 0.9025, [], [9], None, 'wt1', None, None], dtype=object)

        assert bus[10]["index"] == spec[0]
        assert bus[10]["is_root"] == spec[1]
        assert math.isclose(bus[10]["d_P"], spec[2], abs_tol=abs_tol)
        assert math.isclose(bus[10]["d_Q"], spec[3], abs_tol=abs_tol)  
        assert bus[10]["v_max"] == spec[4]
        assert bus[10]["v_min"] == spec[5]
        assert bus[10]["children"] == spec[6]
        assert bus[10]["ancestor"] == spec[7]
        assert bus[10]["generator"] == spec[8]
        assert bus[10]["wind"]['index'] == spec[9]
        assert bus[10]["pv"] == spec[10]
        assert bus[10]["storage"] == spec[11]


# pytest test_bus.py