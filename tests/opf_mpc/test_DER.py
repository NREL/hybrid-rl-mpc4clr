"""
Distributed energy resource (DER) tester.
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

class TestDER:

    def test_wind(self):

        wind = lr_opf_model.Wind

        assert len(wind) == 1

        spec = np.array(['wt1', 11, 0.023113905325443787, 0.01733542899408284, 0.028892381656804734], dtype=object)

        assert wind[0]["index"] == spec[0]
        assert wind[0]["bus_idx"] == spec[1]
        assert math.isclose(wind[0]["w_P_max"], spec[2], abs_tol=abs_tol)
        assert math.isclose(wind[0]["w_Q_max"], spec[3], abs_tol=abs_tol)
        assert math.isclose(wind[0]["w_S_max"], spec[4], abs_tol=abs_tol)

    def test_solar(self):
        
        pv = lr_opf_model.PV

        assert len(pv) == 1

        spec = np.array(['pv1', 13, 0.01733542899408284, 0.013001571745562131, 0.02166928624260355], dtype=object)

        assert pv[0]["index"] == spec[0]
        assert pv[0]["bus_idx"] == spec[1]
        assert math.isclose(pv[0]["p_P_max"], spec[2], abs_tol=abs_tol)
        assert math.isclose(pv[0]["p_Q_max"], spec[3], abs_tol=abs_tol)
        assert math.isclose(pv[0]["p_S_max"], spec[4], abs_tol=abs_tol)

    def test_microturbine(self):

        microturbine = lr_opf_model.Generator

        assert len(microturbine) == 1

        spec = np.array(['g1', 1, 0.023113905325443787, 0.01733542899408284], dtype=object)

        assert microturbine[0]["index"] == spec[0]
        assert microturbine[0]["bus_idx"] == spec[1]
        assert math.isclose(microturbine[0]["g_P_max"], spec[2], abs_tol=abs_tol)
        assert math.isclose(microturbine[0]["g_Q_max"], spec[3], abs_tol=abs_tol)

    def test_battery(self):

        battery = lr_opf_model.Storage

        assert len(battery) == 1

        spec = np.array(['s1', 9, 0.011556952662721894, 0.00866771449704142, 1.0, 0.2, 0.95, 0.9, 1.0], dtype=object)

        assert battery[0]["index"] == spec[0]
        assert battery[0]["bus_idx"] == spec[1]
        assert math.isclose(battery[0]["s_P_max"], spec[2], abs_tol=abs_tol)
        assert math.isclose(battery[0]["s_Q_max"], spec[3], abs_tol=abs_tol)
        assert battery[0]["s_SOC_max"] == spec[4]  
        assert battery[0]["s_SOC_min"] == spec[5]  
        assert battery[0]["s_eff_char"] == spec[6]  
        assert battery[0]["s_eff_dischar"] == spec[7]  
        assert battery[0]["s_cap"] == spec[8]
        
          
# pytest test_DER.py