"""
Load restoration (LR) model tester.
"""

import pytest
import os
import numpy as np
import math

from opf_mpc_envs.envs.lr_opf_model import LROPFModel 

current_dir = os.getcwd()
os.chdir(os.path.dirname(current_dir))

lr_opf_model = LROPFModel()

abs_tol = 1e-04

class TestLRModel:

    def test_load_priority(self):
        
        load_priority_weights_allbus = lr_opf_model.load_priority

        assert len(load_priority_weights_allbus) == 13

        load_priority_weights_loadbus = []
        for i in range(len(load_priority_weights_allbus)):
            if load_priority_weights_allbus[i] != 0.0:
                load_priority_weights_loadbus.append(load_priority_weights_allbus[i])

        assert len(load_priority_weights_loadbus) == 9

        assert load_priority_weights_loadbus == [1.0, 1.0, 0.9, 0.85, 0.8, 0.65, 0.45, 0.4, 0.3]

    def test_restoration_model_solution(self):

        num_bus = 13
        control_horizon = 12                             # 1 hour long with 5 minutes interval
        fuel_reserve = 100                               # in kWh
        soc_reserve = 20                                 # in %
        prior_active_restored_load = [0] * num_bus       # in kW
        prior_reactive_restored_load = [0] * num_bus     # in kW
        mt_remaining_fuel = 1000                         # in kWh
        es_prior_soc = 100                               # in %
        wind_power_forecast = [0] * control_horizon      # in pu
        solar_power_forecast = [0] * control_horizon     # in pu

        solver = 'cbc'

        lr_model = lr_opf_model.build_load_restoration_model(control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                                                             prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                                             wind_power_forecast, solar_power_forecast)

        lr_solution = lr_opf_model.compute_solution(lr_model, solver)

        model_converged, P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, \
        Q_restored, Qmt, Qwtb, Qpvs, Qes, SOC_es, volts = lr_solution

        assert model_converged == True
        
        P_restored_expected = np.array([[115.00000033, 115.00000033, 115.00000033, 115.00000033,
                                         115.00000033, 115.00000033, 115.00000033, 115.00000033,
                                         115.00000033, 115.00000033, 115.00000033, 115.00000033],
                                        [ 85.00000032,  85.00000032,  85.00000032,  85.00000032,
                                            85.00000032,  85.00000032,  85.00000032,  85.00000032,
                                            85.00000032,  85.00000032,  85.00000032,  85.00000032],
                                        [ 49.75000044,  49.75000044,  49.75000044,  49.75000044,
                                            49.75000044,  49.75000044,  49.75000044,  49.75000044,
                                            49.75000044,  49.75000044,  49.75000044,  49.75000044],
                                        [200.00000584, 200.00000584, 200.00000584, 200.00000584,
                                            200.00000584, 200.00000584, 200.00000584, 200.00000584,
                                            200.00000584, 200.00000584, 200.00000584, 200.00000584],
                                        [ 85.00000032,  85.00000032,  85.00000032,  85.00000032,
                                            85.00000032,  85.00000032,  85.00000032,  85.00000032,
                                            85.00000032,  85.00000032,  85.00000032,  85.00000032],
                                        [ 65.24999989,  65.24999989,  65.24999989,  65.24999989,
                                            65.24999989,  65.24999989,  65.24999989,  65.24999989,
                                            65.24999989,  65.24999989,  65.24999989,  65.24999989],
                                        [  0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ],
                                        [  0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ],
                                        [  0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ,
                                            0.        ,   0.        ,   0.        ,   0.        ]])

        for l in range(len(P_restored_expected)):
            Plr_match_check = np.isclose(P_restored[l], P_restored_expected[l], atol=abs_tol)
            assert Plr_match_check.all() == True                            

        Pmt_match_check = np.isclose(Pmt, [400] * control_horizon, atol=abs_tol)
        assert Pmt_match_check.all() == True

        Pes_match_check = np.isclose(Pes, [200] * control_horizon, atol=abs_tol)
        assert Pes_match_check.all() == True

        Pwt_match_check = np.isclose(Pwtb, [0] * control_horizon, atol=abs_tol)
        assert Pwt_match_check.all() == True

        Ppv_match_check = np.isclose(Ppvs, [0] * control_horizon, atol=abs_tol)
        assert Ppv_match_check.all() == True

        SOC_es_expected = [98.715894, 97.431788, 96.147682, 94.86357699999999, 93.579471, 92.29536499999999, \
                            91.011259, 89.727153, 88.44304699999999, 87.158941, 85.874836, 84.59073000000001]

        SOC_match_check = np.isclose(SOC_es, SOC_es_expected, atol=abs_tol)
        assert SOC_match_check.all() == True

        volts_expected = np.array([[1.        , 1.        , 1.        , 1.        , 1.,
                                    1.        , 1.        , 1.        , 1.        , 1.,
                                    1.        , 1.        ],
                                    [0.97778575, 0.97778575, 0.97778575, 0.97778575, 0.97778575,
                                        0.97778575, 0.97778575, 0.97778575, 0.97778575, 0.97778575,
                                        0.97778575, 0.97778575],
                                    [0.97848844, 0.97848844, 0.97848844, 0.97848844, 0.97848844,
                                        0.97848844, 0.97848844, 0.97848844, 0.97848844, 0.97848844,
                                        0.97848844, 0.97848844],
                                    [0.98056121, 0.98056121, 0.98056121, 0.98056121, 0.98056121,
                                        0.98056121, 0.98056121, 0.98056121, 0.98056121, 0.98056121,
                                        0.98056121, 0.98056121],
                                    [0.976641  , 0.976641  , 0.976641  , 0.976641  , 0.976641  ,
                                        0.976641  , 0.976641  , 0.976641  , 0.976641  , 0.976641  ,
                                        0.976641  , 0.976641  ],
                                    [0.9765794 , 0.9765794 , 0.9765794 , 0.9765794 , 0.9765794 ,
                                        0.9765794 , 0.9765794 , 0.9765794 , 0.9765794 , 0.9765794 ,
                                        0.9765794 , 0.9765794 ],
                                    [0.97816145, 0.97816145, 0.97816145, 0.97816145, 0.97816145,
                                        0.97816145, 0.97816145, 0.97816145, 0.97816145, 0.97816145,
                                        0.97816145, 0.97816145],
                                    [0.97889208, 0.97889208, 0.97889208, 0.97889208, 0.97889208,
                                        0.97889208, 0.97889208, 0.97889208, 0.97889208, 0.97889208,
                                        0.97889208, 0.97889208],
                                    [0.97962271, 0.97962271, 0.97962271, 0.97962271, 0.97962271,
                                        0.97962271, 0.97962271, 0.97962271, 0.97962271, 0.97962271,
                                        0.97962271, 0.97962271],
                                    [0.97959151, 0.97959151, 0.97959151, 0.97959151, 0.97959151,
                                        0.97959151, 0.97959151, 0.97959151, 0.97959151, 0.97959151,
                                        0.97959151, 0.97959151],
                                    [0.97794213, 0.97794213, 0.97794213, 0.97794213, 0.97794213,
                                        0.97794213, 0.97794213, 0.97794213, 0.97794213, 0.97794213,
                                        0.97794213, 0.97794213],
                                    [0.97889208, 0.97889208, 0.97889208, 0.97889208, 0.97889208,
                                        0.97889208, 0.97889208, 0.97889208, 0.97889208, 0.97889208,
                                        0.97889208, 0.97889208],
                                    [0.98797839, 0.98797839, 0.98797839, 0.98797839, 0.98797839,
                                        0.98797839, 0.98797839, 0.98797839, 0.98797839, 0.98797839,
                                        0.98797839, 0.98797839]])
    
        for v in range(len(volts_expected)):
            volts_match_check = np.isclose(volts[v], volts_expected[v], atol=abs_tol)
            assert volts_match_check.all() == True 
        
      
# pytest test_load_restoration.py