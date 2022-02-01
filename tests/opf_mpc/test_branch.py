"""
Distribution grid branch/line tester.
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

class TestBranch:

    def test_num_branch(self):

        branch = lr_opf_model.Line

        assert len(branch) == 12
    
    def test_simple_branch(self):

        branch = lr_opf_model.Line

        spec = np.array([7, 3, 8, 0.1313, 0.3856, 2.3239128279566232], dtype=object)
               
        assert branch[6]["index"] == spec[0]
        assert branch[6]["from_node"] == spec[1]
        assert branch[6]["to_node"] == spec[2]
        assert math.isclose(branch[6]["r"], spec[3], abs_tol=abs_tol)
        assert math.isclose(branch[6]["x"], spec[4], abs_tol=abs_tol)
        assert math.isclose(branch[6]["b"], spec[5], abs_tol=abs_tol)
      

# pytest test_branch.py