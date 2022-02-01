"""
Three-phase Optimal Power Flow (OPF) for Critical Load Restoration (lr) Problem in Distribution Grids.
"""

import numpy as np
import pandas as pd
import os
import math
import pyomo.environ as pyo
import warnings

class LROPFModel():       

    def __init__(self):

        # Working and data directories

        self.current_dir = os.getcwd()        
        self.data_path = os.path.dirname(self.current_dir)+"/gym_mpc_env/opf_mpc_envs/envs/data"

        # Parameters

        self.delta_t = 5/60        
        self.load_priority = [0.0, 1.0, 1.0, 0.0, 0.9, 0.85, 0.8, 0.0, 0.65, 0.45, 0.4, 0.3, 0.0]
        self.alpha =  0.2                # $/kWh -- penalty for wind power curtailment
        self.beta = 0.2                  # $/kWh -- penalty for PV power curtailment
        self.psi = 100                   # $/kWh -- penalty (to relax hard constraints)
        self.es_initial_soc = 100        # in %
        self.mt_initial_fuel = 1000      # in kWh

        # Base values

        self.Zbase = 1
        self.Vbase = 4160
        self.Sbase = (self.Vbase**2)/self.Zbase
        self.Cbase = 800
        self.w_to_kW = 1000
        
        # Get grid topology

        self.Bus, self.Line, self.Generator, self.Wind, self.PV, self.Storage = self.get_network_topology()

        # Get grid topology and identify the root bus

        self.check_network_topology()

        # Buses  - where each device is connected to

        self.gen_bus = []
        for g in self.Generator.keys():
            self.gen_bus.append(self.Generator[g]['bus_idx']-1)

        self.wind_bus = []
        for w in self.Wind.keys():
            self.wind_bus.append(self.Wind[w]['bus_idx']-1)

        self.pv_bus = []
        for p in self.PV.keys():
            self.pv_bus.append(self.PV[p]['bus_idx']-1)

        self.stor_bus = []
        for s in self.Storage.keys():
            self.stor_bus.append(self.Storage[s]['bus_idx']-1)

        self.load_bus = []
        for b in self.Bus.keys():
            if self.Bus[b]['d_P'] or self.Bus[b]['d_Q'] > 0:
                self.load_bus.append(b)

        # Root bus - through which the distribution grid is connected with the upstream grid (sub-station)

        self.root_bus = []
        for b in self.Bus.keys():
            if self.Bus[b]['is_root']:
                self.root_bus.append(b)

        # Lines - where each node is connected to

        self.lines_to = {}
        for l in self.Line.keys():
            self.lines_to[self.Line[l]['to_node']] = self.Line[l]

        self.num_buses = len(self.Bus)
        self.num_lines = len(self.lines_to)

        # Voltage (in per unit) at the root bus

        self.v_root = 1.0
            
    def build_load_restoration_model(self, control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                                           prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                           wind_power_forecast, solar_power_forecast):
        """
        Build the load restoration model.
        """

        ## Model

        m = pyo.ConcreteModel()
        
        ## Sets

        m.time_set = pyo.Set(initialize = range(control_horizon))
        m.bus_set = pyo.Set(initialize = self.Bus.keys())
        m.root_set = pyo.Set(initialize = self.root_bus)        
        m.gen_set = pyo.Set(initialize = self.gen_bus)
        m.wind_set = pyo.Set(initialize = self.wind_bus)
        m.pv_set = pyo.Set(initialize = self.pv_bus)
        m.stor_set = pyo.Set(initialize = self.stor_bus)
        m.load_set = pyo.Set(initialize = self.load_bus)
            
        ## Variables

        m.v = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)     # voltage square

        m.fp = pyo.Var(m.bus_set, m.time_set, within=pyo.Reals)               # active power flow
        m.fq = pyo.Var(m.bus_set, m.time_set, within=pyo.Reals)               # reactive power flow

        m.gp = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)    # gen active power generation
        m.gq = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)    # gen reactive power generation

        m.spc = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # storage active power charging
        m.spd = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # storage active power discharging
        m.sqc = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # storage reactive power absorption        
        m.sqd = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # storage reactive power injection        
        m.ssoc = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)  # storage state of charge
        m.sc = pyo.Var(m.bus_set, m.time_set, within=pyo.Binary)              # storage charging indicator
        m.sd = pyo.Var(m.bus_set, m.time_set, within=pyo.Binary)              # storage discharging indicator

        m.wdp = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # wind active power output 
        m.wdq = pyo.Var(m.bus_set, m.time_set, within=pyo.Reals)              # wind reactive injection/absorption
        m.wdpc = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)  # wind power curtailment

        m.pvp = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # PV active power output 
        m.pvq = pyo.Var(m.bus_set, m.time_set, within=pyo.Reals)              # PV reactive injection/absorption
        m.pvpc = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)  # PV power curtailment

        m.rlp = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # active restored load 
        m.rlq = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # reactive restored load 

        m.mup = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # slack var for hard constraint relaxation
        m.muq = pyo.Var(m.bus_set, m.time_set, within=pyo.NonNegativeReals)   # slack var for hard constraint relaxation

        ## Parameters

        # Generator
        m.gp_max = pyo.Param(m.gen_set, within=pyo.NonNegativeReals, \
            initialize={self.Generator[g]['bus_idx']-1 : self.Generator[g]['g_P_max'] for g in self.Generator.keys()})
        m.gq_max = pyo.Param(m.gen_set, within=pyo.NonNegativeReals, \
            initialize={self.Generator[g]['bus_idx']-1 : self.Generator[g]['g_Q_max'] for g in self.Generator.keys()})
        m.gen_pf = pyo.Param(m.gen_set, within=pyo.NonNegativeReals, initialize=0.75)
        m.fuel_reserve = pyo.Param(m.gen_set, mutable=True, within=pyo.NonNegativeReals, initialize=fuel_reserve)
        m.mt_remaining_fuel = pyo.Param(m.gen_set, mutable=True, within=pyo.NonNegativeReals, initialize=mt_remaining_fuel)

        # Wind
        wind_upperbound_dict = {(i,j): wind_power_forecast[j] for i in m.wind_set for j in m.time_set} 
        m.wind_upperbound = pyo.Param(m.wind_set, m.time_set, initialize=wind_upperbound_dict, mutable = True)
        m.wind_s_max = pyo.Param(m.wind_set, within=pyo.NonNegativeReals, \
            initialize={self.Wind[w]['bus_idx']-1 : self.Wind[w]['w_S_max'] for w in self.Wind.keys()})

        # PV
        pv_upperbound_dict = {(i,j): solar_power_forecast[j] for i in m.pv_set for j in m.time_set} 
        m.pv_upperbound = pyo.Param(m.pv_set, m.time_set, initialize=pv_upperbound_dict, mutable = True)
        m.pv_s_max = pyo.Param(m.pv_set, within=pyo.NonNegativeReals, \
            initialize={self.PV[p]['bus_idx']-1 : self.PV[p]['p_S_max'] for p in self.PV.keys()})

        # Storage
        m.spc_max = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_P_max'] for s in self.Storage.keys()})
        m.spd_max = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_P_max'] for s in self.Storage.keys()})
        m.sqc_max = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_Q_max'] for s in self.Storage.keys()})
        m.sqd_max = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_Q_max'] for s in self.Storage.keys()})
        m.ssoc_min = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_SOC_min'] for s in self.Storage.keys()})
        m.ssoc_max = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_SOC_max'] for s in self.Storage.keys()})
        m.initial_soc = pyo.Param(m.stor_set, mutable=True, within=pyo.NonNegativeReals, initialize=es_prior_soc/100)
        m.soc_reserve = pyo.Param(m.stor_set, mutable=True, within=pyo.NonNegativeReals, initialize=soc_reserve/100)
        m.charging_eff = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_eff_char'] for s in self.Storage.keys()})
        m.discharging_eff = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_eff_dischar'] for s in self.Storage.keys()})
        m.storage_capacity = pyo.Param(m.stor_set, within=pyo.NonNegativeReals, \
            initialize={self.Storage[s]['bus_idx']-1 : self.Storage[s]['s_cap'] for s in self.Storage.keys()})  

        # Load
       
        restored_activepower_upperbound_dict = {i: self.Bus[i]['d_P'] for i in m.load_set} 
        m.restored_activepower_upperbound = pyo.Param(m.load_set, initialize=restored_activepower_upperbound_dict, mutable = True)    
        restored_reactivepower_upperbound_dict = {i: self.Bus[i]['d_Q'] for i in m.load_set} 
        m.restored_reactivepower_upperbound = pyo.Param(m.load_set, initialize=restored_reactivepower_upperbound_dict, mutable = True) 
        prior_restored_activepower_dict = {i: prior_active_restored_load[i] for i in m.load_set} 
        m.prior_restored_activepower = pyo.Param(m.load_set, initialize=prior_restored_activepower_dict, mutable = True) 
        prior_restored_reactivepower_dict = {i: prior_reactive_restored_load[i] for i in m.load_set} 
        m.prior_restored_reactivepower = pyo.Param(m.load_set, initialize=prior_restored_reactivepower_dict, mutable = True)

        ## Objective function
            # maximize the total /priority-weighted/ load (active power) served
            # penalize prior restored load shedding
            # penalize wind and pv power curtailment

        def objective_fun(m):            
            
            return sum(self.load_priority[b]*m.rlp[b,t]*self.delta_t for b in m.bus_set for t in m.time_set) \
                    - sum(self.load_priority[b]*m.mup[b,t]*self.psi*self.delta_t for b in m.bus_set for t in m.time_set) \
                        - sum(m.wdpc[b,t]*self.alpha*self.delta_t for b in m.bus_set for t in m.time_set) \
                            - sum(m.pvpc[b,t]*self.beta*self.delta_t for b in m.bus_set for t in m.time_set)   

        m.restoration = pyo.Objective(rule = objective_fun,sense = pyo.maximize)

        ## Constraints

        # 1. all buses with generators
               
        m.gp_upperbound_constraint = pyo.Constraint(m.gen_set, m.time_set)
        m.gq_upperbound_constraint = pyo.Constraint(m.gen_set, m.time_set)
        m.gp_fuel_usage_constraint = pyo.Constraint(m.gen_set)
        m.gq_fuel_usage_constraint = pyo.Constraint(m.gen_set)
        m.gen_fuel_reserve_constraint = pyo.Constraint(m.gen_set)

        # Power bounds
        for g in m.gen_set:
            for t in m.time_set:                              
                m.gp_upperbound_constraint[g,t] = (None, m.gp[g,t] - m.gp_max[g], 0)
                m.gq_upperbound_constraint[g,t] = (None, m.gq[g,t] - m.gq_max[g], 0)

        # Fuel usage 
        for g in m.gen_set:
            m.gp_fuel_usage_constraint[g] = \
                (None, sum(m.gp[g,t]*self.delta_t*self.Sbase/self.w_to_kW for t in m.time_set) - m.mt_remaining_fuel[g], 0)
            m.gq_fuel_usage_constraint[g] = \
                (None, sum(m.gq[g,t]*self.delta_t*self.Sbase/self.w_to_kW for t in m.time_set) - m.mt_remaining_fuel[g]*m.gen_pf[g], 0)

        # End-of-horizon fuel reserve
        for g in m.gen_set:
            m.gen_fuel_reserve_constraint[g] = \
                (0, m.mt_remaining_fuel[g] - sum(m.gp[g,t]*self.delta_t*self.Sbase/self.w_to_kW for t in m.time_set) - m.fuel_reserve[g], None)
       
        # 2. all buses with wind turbines

        m.wdp_upperbound_constraint = pyo.Constraint(m.wind_set, m.time_set)
        m.wdpc_upperbound_constraint = pyo.Constraint(m.wind_set, m.time_set)
        m.wdq_upperbound_constraint = pyo.Constraint(m.wind_set, m.time_set)
        m.wdq_lowerbound_constraint = pyo.Constraint(m.wind_set, m.time_set)

        # Power output and curtailment bounds
        for w in m.wind_set:
            for t in m.time_set:
                m.wdp_upperbound_constraint[w,t] = (m.wdp[w,t] == m.wind_upperbound[w,t])
                m.wdpc_upperbound_constraint[w,t] = (None, m.wdpc[w,t] - m.wind_upperbound[w,t], 0)
                m.wdq_upperbound_constraint[w,t] = (None, m.wdq[w,t] - np.sqrt(m.wind_s_max[w]**2 - m.wind_upperbound[w,t]._value**2), 0)
                m.wdq_lowerbound_constraint[w,t] = (0, m.wdq[w,t] + np.sqrt(m.wind_s_max[w]**2 - m.wind_upperbound[w,t]._value**2), None)
        
        # 3. all buses with PVs

        m.pvp_upperbound_constraint = pyo.Constraint(m.pv_set, m.time_set)
        m.pvpc_upperbound_constraint = pyo.Constraint(m.pv_set, m.time_set)
        m.pvq_upperbound_constraint = pyo.Constraint(m.pv_set, m.time_set)
        m.pvq_lowerbound_constraint = pyo.Constraint(m.pv_set, m.time_set)

        # Power output and curtailment bounds
        for p in m.pv_set:
            for t in m.time_set:
                m.pvp_upperbound_constraint[p,t] = (m.pvp[p,t] == m.pv_upperbound[p,t])
                m.pvpc_upperbound_constraint[p,t] = (None, m.pvpc[p,t] - m.pv_upperbound[p,t], 0)
                m.pvq_upperbound_constraint[p,t] = (None, m.pvq[p,t] - np.sqrt(m.pv_s_max[p]**2 - m.pv_upperbound[p,t]._value**2), 0)
                m.pvq_lowerbound_constraint[p,t] = (0, m.pvq[w,t] + np.sqrt(m.pv_s_max[p]**2 - m.pv_upperbound[p,t]._value**2), None)
              
        # 4. all buses with storages

        m.spc_upperbound_constraint = pyo.Constraint(m.stor_set, m.time_set)
        m.spd_upperbound_constraint = pyo.Constraint(m.stor_set, m.time_set)
        m.sqc_upperbound_constraint = pyo.Constraint(m.stor_set, m.time_set)
        m.sqd_upperbound_constraint = pyo.Constraint(m.stor_set, m.time_set)
        m.storage_charge_discharge_complementarity = pyo.Constraint(m.stor_set, m.time_set)
        m.ssoc_lowerbound_constraint = pyo.Constraint(m.stor_set, m.time_set)            
        m.ssoc_upperbound_constraint = pyo.Constraint(m.stor_set, m.time_set)        
        m.storage_soc_dynamics = pyo.Constraint(m.stor_set, m.time_set)        
        m.storage_soc_reserve_constraint = pyo.Constraint(m.stor_set)      
    
        for s in m.stor_set:
            for t in m.time_set:
                # Charging power bounds               
                m.spc_upperbound_constraint[s,t] = \
                        (None, m.spc[s,t] - m.spc_max[s]*m.sc[s,t], 0)
                m.sqc_upperbound_constraint[s,t] = \
                        (None, m.sqc[s,t] - m.sqc_max[s]*m.sc[s,t], 0)               
                
                # Discharging power bounds                
                m.spd_upperbound_constraint[s,t] = \
                        (None, m.spd[s,t] - m.spd_max[s]*m.sd[s,t], 0)
                m.sqd_upperbound_constraint[s,t] = \
                        (None, m.sqd[s,t] - m.sqd_max[s]*m.sd[s,t], 0)
                                
                # Avoid simultaneous charge and discharge
                m.storage_charge_discharge_complementarity[s,t] = (m.sc[s,t] + m.sd[s,t] <= 1)

                # SOC bounds
                m.ssoc_lowerbound_constraint[s,t] = (0 , m.ssoc[s,t] - m.ssoc_min[s], None)
                m.ssoc_upperbound_constraint[s,t] = (None , m.ssoc[s,t] - m.ssoc_max[s], 0)

                # SOC dynamics                
                if t == m.time_set.first():
                    m.storage_soc_dynamics[s,t] = (m.ssoc[s,t] == m.initial_soc[s] \
                        + (m.charging_eff[s]*m.spc[s,t]/m.storage_capacity[s]) \
                            - m.spd[s,t]/(m.discharging_eff[s]*m.storage_capacity[s]))

                if t > m.time_set.first():
                    m.storage_soc_dynamics[s,t] = (m.ssoc[s,t] == m.ssoc[s,t-1] + (m.charging_eff[s]*m.spc[s,t]/m.storage_capacity[s]) \
                        - m.spd[s,t]/(m.discharging_eff[s]*m.storage_capacity[s]))                
                
                # End-of-horizon SOC reserve
                if t == m.time_set.last():
                    m.storage_soc_reserve_constraint[s] = (0 , m.ssoc[s,t] - m.soc_reserve[s], None)

        # 5. all buses with loads

        m.rlp_upperbound_constraint = pyo.Constraint(m.load_set, m.time_set)
        m.rlq_upperbound_constraint = pyo.Constraint(m.load_set, m.time_set)
        m.load_power_factor_constraint = pyo.Constraint(m.load_set, m.time_set)
        m.prior_active_restored_load_shedding_constraint = pyo.Constraint(m.load_set, m.time_set)
        m.prior_reactive_restored_load_shedding_constraint = pyo.Constraint(m.load_set, m.time_set)
               
        for l in m.load_set:
            for t in m.time_set:
                # Restored load feasibility range
                m.rlp_upperbound_constraint[l,t] = (None , m.rlp[l,t] - m.restored_activepower_upperbound[l], 0)
                m.rlq_upperbound_constraint[l,t] = (None , m.rlq[l,t] - m.restored_reactivepower_upperbound[l], 0)

                # Power factor constraint: Qres/Pres = Qdem/Pdem
                m.load_power_factor_constraint[l,t] = (m.rlq[l,t]*m.restored_activepower_upperbound[l] == \
                                                        m.rlp[l,t]*m.restored_reactivepower_upperbound[l])

                # Relaxation
                if t == m.time_set.first():
                    m.prior_active_restored_load_shedding_constraint[l,t] = \
                        (None , m.prior_restored_activepower[l] - m.rlp[l,t] - m.mup[l,t], 0)
                    m.prior_reactive_restored_load_shedding_constraint[l,t] = \
                        (None , m.prior_restored_reactivepower[l] - m.rlq[l,t] - m.muq[l,t], 0)
                if t > m.time_set.first():
                    m.prior_active_restored_load_shedding_constraint[l,t] = \
                        (None , m.rlp[l,t-1] - m.rlp[l,t] - m.mup[l,t], 0)
                    m.prior_reactive_restored_load_shedding_constraint[l,t] = \
                        (None , m.rlq[l,t-1] - m.rlq[l,t] - m.muq[l,t], 0)

        # 6. all buses without generator

        m.gp_non_gen_bus_constraint = pyo.Constraint((m.bus_set - m.gen_set), m.time_set)
        m.gq_non_gen_bus_constraint = pyo.Constraint((m.bus_set - m.gen_set), m.time_set)

        for b in (m.bus_set - m.gen_set):
            for t in m.time_set:
                m.gp_non_gen_bus_constraint[b,t] = (m.gp[b,t] == 0.0)
                m.gq_non_gen_bus_constraint[b,t] = (m.gq[b,t] == 0.0)

        # 7. all buses without wind

        m.wdp_non_wind_bus_constraint = pyo.Constraint((m.bus_set - m.wind_set), m.time_set)
        m.wdpc_non_wind_bus_constraint = pyo.Constraint((m.bus_set - m.wind_set), m.time_set)
        m.wdq_non_wind_bus_constraint = pyo.Constraint((m.bus_set - m.wind_set), m.time_set)

        for b in (m.bus_set - m.wind_set):
            for t in m.time_set:
                m.wdp_non_wind_bus_constraint[b,t] = (m.wdp[b,t] == 0.0)
                m.wdpc_non_wind_bus_constraint[b,t] = (m.wdpc[b,t] == 0.0)
                m.wdq_non_wind_bus_constraint[b,t] = (m.wdq[b,t] == 0.0)               

        # 8. all buses without PV

        m.pvp_non_pv_bus_constraint = pyo.Constraint((m.bus_set - m.pv_set), m.time_set)
        m.pvpc_non_pv_bus_constraint = pyo.Constraint((m.bus_set - m.pv_set), m.time_set)
        m.pvq_non_pv_bus_constraint = pyo.Constraint((m.bus_set - m.pv_set), m.time_set)

        for b in (m.bus_set - m.pv_set):
            for t in m.time_set:
                m.pvp_non_pv_bus_constraint[b,t] = (m.pvp[b,t] == 0.0)
                m.pvpc_non_pv_bus_constraint[b,t] = (m.pvpc[b,t] == 0.0)
                m.pvq_non_pv_bus_constraint[b,t] = (m.pvq[b,t] == 0.0)             
        
        # 9. all buses without storage

        m.spc_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)
        m.spd_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set) 
        m.sqc_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)
        m.sqd_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)       
        m.ssoc_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)
        m.sc_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)
        m.sd_non_storage_bus_constraint = pyo.Constraint((m.bus_set - m.stor_set), m.time_set)
      
        for b in (m.bus_set - m.stor_set):
            for t in m.time_set:
                m.spc_non_storage_bus_constraint[b,t] = (m.spc[b,t] == 0)
                m.spd_non_storage_bus_constraint[b,t] = (m.spd[b,t] == 0)                
                m.sqc_non_storage_bus_constraint[b,t] = (m.sqc[b,t] == 0)
                m.sqd_non_storage_bus_constraint[b,t] = (m.sqd[b,t] == 0)                
                m.ssoc_non_storage_bus_constraint[b,t] = (m.ssoc[b,t] == 0)
                m.sc_non_storage_bus_constraint[b,t] = (m.sc[b,t] == 0)
                m.sd_non_storage_bus_constraint[b,t] = (m.sd[b,t] == 0)             

        # 10. all buses without load

        m.rlp_non_load_bus_constraint = pyo.Constraint((m.bus_set - m.load_set), m.time_set)
        m.rlq_non_load_bus_constraint = pyo.Constraint((m.bus_set - m.load_set), m.time_set)
        m.mup_non_load_bus_constraint = pyo.Constraint((m.bus_set - m.load_set), m.time_set)
        m.muq_non_load_bus_constraint = pyo.Constraint((m.bus_set - m.load_set), m.time_set)

        for b in (m.bus_set - m.load_set):
            for t in m.time_set:
                m.rlp_non_load_bus_constraint[b,t] = (m.rlp[b,t] == 0)
                m.rlq_non_load_bus_constraint[b,t] = (m.rlq[b,t] == 0)
                m.mup_non_load_bus_constraint[b,t] = (m.mup[b,t] == 0)
                m.muq_non_load_bus_constraint[b,t] = (m.muq[b,t] == 0)

        # 11. nodal power balance equations (LinDistFlow) and voltage bounds

        m.active_power_balance_equation = pyo.Constraint(m.bus_set, m.time_set)
        m.reactive_power_balance_equation = pyo.Constraint(m.bus_set, m.time_set)
        m.nodal_voltage_lowerbound_constraint = pyo.Constraint(m.bus_set, m.time_set)
        m.nodal_voltage_upperbound_constraint = pyo.Constraint(m.bus_set, m.time_set)

        for b in m.bus_set:
            for t in m.time_set:
                m.active_power_balance_equation[b,t] = \
                    (m.rlp[b,t] - m.gp[b,t] - (m.wdp[b,t]-m.wdpc[b,t]) - (m.pvp[b,t]-m.pvpc[b,t]) + m.spc[b,t] - m.spd[b,t] \
                        + sum(m.fp[k,t] for k in self.Bus[b]['children']) == m.fp[b,t])
                
                m.reactive_power_balance_equation[b,t] = \
                    (m.rlq[b,t] - m.gq[b,t] - m.wdq[b,t] - m.pvq[b,t] + m.sqc[b,t] - m.sqd[b,t] \
                        + sum(m.fq[k,t] for k in self.Bus[b]['children']) == m.fq[b,t])

                m.nodal_voltage_lowerbound_constraint[b,t] = (0, m.v[b,t] - self.Bus[b]['v_min'], None)
                m.nodal_voltage_upperbound_constraint[b,t] = (None, m.v[b,t] - self.Bus[b]['v_max'], 0)

        # 12. Voltage equation (LinDistFlow)

        m.voltage_equation = pyo.Constraint((m.bus_set - m.root_set), m.time_set) 

        for b in (m.bus_set - m.root_set):
            for t in m.time_set:
                b_ancestor = self.Bus[b]['ancestor'][0]
                m.voltage_equation[b,t] = (m.v[b,t] == m.v[b_ancestor,t] - 2*(self.lines_to[b]['r'] * m.fp[b,t] + self.lines_to[b]['x'] * m.fq[b,t]))

        # 13. Voltage and power flow constraints for root node(s)

        m.rootbus_voltage_equation = pyo.Constraint(m.root_set, m.time_set)
        m.rootbus_active_power_balance_equation = pyo.Constraint(m.root_set, m.time_set)
        m.rootbus_reactive_power_balance_equation = pyo.Constraint(m.root_set, m.time_set)

        for b in m.root_set:
            for t in m.time_set:
                m.rootbus_voltage_equation[b,t] = (m.v[b,t] == self.v_root)
                m.rootbus_active_power_balance_equation[b,t] = (m.fp[b,t] == 0)
                m.rootbus_reactive_power_balance_equation[b,t] = (m.fq[b,t] == 0)

        return m       
    
    def solve_model(self, model, solver, tee, solver_options):
        """
        Solve the load restoration optimization model.
        """

        solver = pyo.SolverFactory(solver)

        if solver_options:
            for k, v in solver_options.items():
                solver.options[k] = v

        results = solver.solve(model, tee=tee)

        if results.solver.termination_condition == pyo.TerminationCondition.optimal and \
            results.solver.status == pyo.SolverStatus.ok:
                opf_converged = True
        else:
            opf_converged = False
            
        return opf_converged
    
    def compute_solution(self, model, solver, solver_tee=False, solver_options=None):
        """
        Compute the solution of the optimization model.
        """
        
        opf_converged = self.solve_model(model, solver, solver_tee, solver_options)

        if opf_converged:
            
            # Microturbine
            mt_p = {mt: [pyo.value(model.gp[mt,t])*self.Sbase/self.w_to_kW for t in model.time_set] for mt in model.gen_set}
            mt_q = {mt: [pyo.value(model.gq[mt,t])*self.Sbase/self.w_to_kW for t in model.time_set] for mt in model.gen_set}

            # Wind
            wt_p = {wt: [pyo.value(model.wdp[wt,t])*self.Sbase/self.w_to_kW for t in model.time_set] for wt in model.wind_set}
            wt_pc = {wt: [pyo.value(model.wdpc[wt,t])*self.Sbase/self.w_to_kW for t in model.time_set] for wt in model.wind_set}
            wt_q = {wt: [pyo.value(model.wdq[wt,t])*self.Sbase/self.w_to_kW for t in model.time_set] for wt in model.wind_set}

            # PV
            pv_p = {pv: [pyo.value(model.pvp[pv,t])*self.Sbase/self.w_to_kW for t in model.time_set] for pv in model.pv_set}
            pv_pc = {pv: [pyo.value(model.pvpc[pv,t])*self.Sbase/self.w_to_kW for t in model.time_set] for pv in model.pv_set}
            pv_q = {pv: [pyo.value(model.pvq[pv,t])*self.Sbase/self.w_to_kW for t in model.time_set] for pv in model.pv_set}

            # Storage            
            st_p = {st: [(pyo.value(model.spd[st,t]) - pyo.value(model.spc[st,t]))*self.Sbase/self.w_to_kW \
                for t in model.time_set] for st in model.stor_set}
            st_q = {st: [(pyo.value(model.sqd[st,t]) - pyo.value(model.sqc[st,t]))*self.Sbase/self.w_to_kW \
                for t in model.time_set] for st in model.stor_set}
            st_soc = {st: [pyo.value(model.ssoc[st,t])*100 for t in model.time_set] for st in model.stor_set}

            # Load
            ld_p = {ld: [pyo.value(model.rlp[ld,t])*self.Sbase/self.w_to_kW for t in model.time_set] for ld in model.load_set}
            ld_q = {ld: [pyo.value(model.rlq[ld,t])*self.Sbase/self.w_to_kW for t in model.time_set] for ld in model.load_set}

            # Bus voltages
            voltages = {b: [pyo.value(model.v[b,t]) for t in model.time_set] for b in model.bus_set}

            # Line flows
            flows_p = {b: [pyo.value(model.fp[b,t])*self.Sbase/self.w_to_kW for t in model.time_set] for b in model.bus_set}
            flows_q = {b: [pyo.value(model.fq[b,t])*self.Sbase/self.w_to_kW for t in model.time_set] for b in model.bus_set}

            # Re-organize the results in numpy arrays and lists - for gym conveniency

            P_restored = np.array(list(ld_p.values()))
            Pmt = list(mt_p.values())[0]
            Pes = list(st_p.values())[0]
            Pwtb = list(wt_p.values())[0]
            Pwt_cut = list(wt_pc.values())[0]
            Ppvs = list(pv_p.values())[0]
            Ppv_cut = list(pv_pc.values())[0]
            SOC_es = list(st_soc.values())[0]
            Q_restored = np.array(list(ld_q.values())) 
            Qmt = list(mt_q.values())[0] 
            Qwtb = list(wt_q.values())[0]  
            Qpvs = list(pv_q.values())[0] 
            Qes = list(st_q.values())[0]  
            volts = np.array(list(voltages.values()))

        else:
                       
            P_restored = []
            Pmt = []
            Pes = []
            Pwtb = []
            Pwt_cut = []
            Ppvs = []
            Ppv_cut = [] 
            SOC_es = []
            Q_restored = [] 
            Qmt = []
            Qwtb = [] 
            Qpvs = [] 
            Qes = [] 
            volts = [] 

        return opf_converged, \
               P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, \
               Q_restored, Qmt, Qwtb, Qpvs, Qes, \
               SOC_es, volts
      
    def get_network_topology(self):
        """
        Get the network topology.
        """

        # Buses

        Bus = {}

        nodes_raw = pd.read_csv(self.data_path+'/network_data/13bus/nodes.csv')

        for n in range(len(nodes_raw)):
            
            bus_params = {'index': None, 
                          'is_root': None,
                          'd_P': None,
                          'd_Q': None,
                          'cosphi': None,
                          'tanphi': None,
                          'v_max': None,
                          'v_min': None,
                          'children': [],
                          'ancestor': [],
                          'generator': None,
                          'wind': None,
                          'pv': None,
                          'storage': None,               
                         }

            bus_params['index'] = nodes_raw['index'][n]
            bus_params['d_P'] = nodes_raw['d_P'][n]*self.w_to_kW/self.Sbase
            bus_params['d_Q'] = nodes_raw['d_Q'][n]*self.w_to_kW/self.Sbase
            bus_params['v_max'] = nodes_raw['v_max'][n]
            bus_params['v_min'] = nodes_raw['v_min'][n]
            
            bus_params['is_root'] = False
            
            bus_params['children'] = []
            bus_params['ancestor'] = []
                        
            if bus_params['d_P'] and bus_params['d_Q'] !=0:
                bus_params['cosphi'] = bus_params['d_P']/math.sqrt((bus_params['d_P'])**2 + (bus_params['d_Q'])**2)
                bus_params['tanphi'] = math.tan(math.acos(bus_params['cosphi']))
            else:
                bus_params['cosphi'] = 0.0
                bus_params['tanphi'] = 0.0
                    
            Bus[n] = bus_params

        # Lines

        Line = {}

        lines_raw = pd.read_csv(self.data_path+'/network_data/13bus/lines.csv')

        for l in range(len(lines_raw)):
            
            line_params = {'index': None, 
                           'from_node': None,
                           'to_node': None,                   
                           'r': None,
                           'x': None,
                           'b': None,
                           's_max': None              
                          }

            line_params['index'] = lines_raw['index'][l]
            line_params['from_node'] = lines_raw['from_node'][l]-1
            line_params['to_node'] = lines_raw['to_node'][l]-1
            line_params['r'] = lines_raw['r'][l]/self.Zbase
            line_params['x'] = lines_raw['x'][l]/self.Zbase
            line_params['b'] = line_params['x']/((line_params['r'])**2+(line_params['x'])**2)
            line_params['s_max'] = lines_raw['s_max'][l]/self.Sbase
            
            Line[l] = line_params
            
            Bus[line_params['from_node']]['children'].append(line_params['to_node'])
            Bus[line_params['to_node']]['ancestor'].append(line_params['from_node'])

        # Generators (non-renewable gens such as microturbine)

        Generator = {}

        generators_raw = pd.read_csv(self.data_path+'/network_data/13bus/generators.csv')        

        for g in range(len(generators_raw)):
            
            gen_params = {'index': None, 
                          'bus_idx': None,
                          'g_P_max': None,
                          'g_Q_max': None,
                          'cost': None                 
                         }
            
            gen_params['index'] = generators_raw['index'][g]
            gen_params['bus_idx'] = generators_raw['node'][g]
            gen_params['g_P_max'] = generators_raw['p_max'][g]*self.w_to_kW/self.Sbase
            g_S_max =  generators_raw['s_max'][g]*self.w_to_kW/self.Sbase
            gen_params['g_Q_max'] = math.sqrt((g_S_max)**2 - (gen_params['g_P_max'])**2)
            gen_params['cost'] = generators_raw['cost'][g]
            
            Generator[g] = gen_params
            
            Bus[gen_params['bus_idx']-1]['generator'] = gen_params

        # Windturbines

        Wind = {}

        windturbines_raw = pd.read_csv(self.data_path+'/network_data/13bus/windturbines.csv')        

        for w in range(len(windturbines_raw)):
            
            wind_params = {'index': None, 
                           'bus_idx': None,
                           'w_P_max': None,
                           'w_Q_max': None,
                           'w_S_max': None,             
                          }
            
            wind_params['index'] = windturbines_raw['index'][w]
            wind_params['bus_idx'] = windturbines_raw['node'][w]
            wind_params['w_P_max'] = windturbines_raw['p_max'][w]*self.w_to_kW/self.Sbase
            wind_params['w_S_max'] = windturbines_raw['s_max'][w]*self.w_to_kW/self.Sbase   
            wind_params['w_Q_max'] = math.sqrt((wind_params['w_S_max'])**2 - (wind_params['w_P_max'])**2)
            
            Wind[w] = wind_params
            
            Bus[wind_params['bus_idx']-1]['wind'] = wind_params

        # Photovoltaics (PVs)

        PV = {}

        pvs_raw = pd.read_csv(self.data_path+'/network_data/13bus/pvs.csv')

        for p in range(len(pvs_raw)):
            
            pv_params = {'index': None, 
                         'bus_idx': None,
                         'p_P_max': None,
                         'p_Q_max': None,
                         'p_S_max': None,             
                        }
            
            pv_params['index'] = pvs_raw['index'][p]
            pv_params['bus_idx'] = pvs_raw['node'][p]
            pv_params['p_P_max'] = pvs_raw['p_max'][p]*self.w_to_kW/self.Sbase    
            pv_params['p_S_max'] = pvs_raw['s_max'][p]*self.w_to_kW/self.Sbase 
            pv_params['p_Q_max'] = math.sqrt((pv_params['p_S_max'])**2 - (pv_params['p_P_max'])**2)
            
            PV[p] = pv_params
            
            Bus[pv_params['bus_idx']-1]['pv'] = pv_params

        # Energy storages

        Storage = {}

        storages_raw = pd.read_csv(self.data_path+'/network_data/13bus/storages.csv')

        for s in range(len(storages_raw)):
            
            storage_params = {'index': None, 
                    'bus_idx': None,
                    's_P_max': None,
                    's_Q_max': None,
                    's_SOC_max': None,
                    's_SOC_min': None,
                    's_eff_char': None,
                    's_eff_dischar': None,
                    's_cap': None                      
                    }
            
            storage_params['index'] = storages_raw['index'][s]
            storage_params['bus_idx'] = storages_raw['node'][s]
            storage_params['s_P_max'] = storages_raw['p_max'][s]*self.w_to_kW/self.Sbase
            s_S_max = storages_raw['s_max'][s]*self.w_to_kW/self.Sbase
            storage_params['s_Q_max'] = math.sqrt((s_S_max)**2 - (storage_params['s_P_max'])**2)  
            storage_params['s_SOC_max'] = storages_raw['SOC_max'][s]/100
            storage_params['s_SOC_min'] = storages_raw['SOC_min'][s]/100
            storage_params['s_eff_char'] = storages_raw['eff_char'][s]/100
            storage_params['s_eff_dischar'] = storages_raw['eff_dischar'][s]/100
            storage_params['s_cap'] = storages_raw['capacity'][s]/self.Cbase
            
            Storage[s] = storage_params
            
            Bus[storage_params['bus_idx']-1]['storage'] = storage_params
            
        return Bus, Line, Generator, Wind, PV, Storage

    def check_network_topology(self):
        """
        Check the network topology.
        """

        r = 0
        root_bus = None

        for b in self.Bus.keys():
            
            l = len(self.Bus[b]['ancestor'])
            
            if l > 1:
                warnings.warn('Network Not Radial; Bus ' f"{self.Bus[b]['index']}")
            elif l == 0:
                self.Bus[b]['is_root'] = True
                root_bus = b
                r += 1                
                
        if r == 0:
            warnings.warn("No root detected")
            root_bus = None
        elif r > 1:
            warnings.warn("More than one root detected")