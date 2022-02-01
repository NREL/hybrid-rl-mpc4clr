"""
Distribution grid load tester.
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

class TestLoad:

    def test_num_load(self):

        bus = lr_opf_model.Bus

        num_load = 0

        for idx in range(len(bus)):
            pl = bus[idx]["d_P"]
            ql = bus[idx]["d_Q"]
            if pl != 0.0 or ql != 0.0:
                num_load += 1

        assert num_load == 9

    def test_simple_load(self):

        bus = lr_opf_model.Bus

        spec = np.array([5, 0.002874791974852071, 0.0016757581360946747, 0.8639363184095202], dtype=object)

        assert bus[4]["index"] == spec[0]
        assert math.isclose(bus[4]["d_P"], spec[1], abs_tol=abs_tol)
        assert math.isclose(bus[4]["d_Q"], spec[2], abs_tol=abs_tol)
        assert math.isclose(bus[4]["cosphi"], spec[3], abs_tol=abs_tol)

          
# pytest test_load.py