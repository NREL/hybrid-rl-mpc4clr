"""
Three-phase Optimal Power Flow (OPF) for Critical Load Restoration (lr) Problem in Distribution Grids.
"""

import pandas as pd 
import numpy as np
from datetime import datetime
import os

import julia
from julia.api import Julia
jl = Julia(compiled_modules=False)  
jl = julia.Julia()

class OPFJuMP():

    def __init__(self):
        
        self.cur_dir = os.getcwd()
        self.DATA_DIR = "basecase"    # default/base network data (if any)

        self.num_buses = 13
        self.load_buses = [2, 3, 5, 6, 7, 9, 10, 11, 12]
        self.non_load_buses = [0, 1, 4, 8]
        
        self.num_time_steps = 72     # optimization horizon (6 hours == 72 5-minutes)
        
        self.network = self.network_topology("13bus")
        
        self.load_power_forecast = self.get_load_power_forecast(self.network['buses'])

    def network_topology(self, distribution_system):
        """ 
        Get the distribution network components. 
        """
        
        #jl.include("data_handler_threephase.jl")   # Distribution network data preprocessor written in Julia # Absolute pathe
        jl.include(os.path.dirname(self.cur_dir) + "/gym_opf_env/opf_envs/envs/data_handler_threephase.jl")   # Relative path from the agent training scipt

        BUSES, LINES, GENERATORS, WINDTURBINES, PVS, STORAGES = jl.load_case_data(datafile=distribution_system)

        network_topology = {'buses': BUSES, 
                            'lines': LINES, 
                            'generators': GENERATORS, 
                            'windturbines': WINDTURBINES, 
                            'pvs': PVS, 
                            'storages': STORAGES}

        return network_topology

    def get_load_power_forecast(self, buses):
        """
        Get demanded load power forecasts at each control step.
        """ 

        d_P = np.zeros((len(buses), self.num_time_steps))
        d_Q = np.zeros((len(buses), self.num_time_steps))

        for b in jl.keys(buses):
            for t in range(self.num_time_steps):
                if b == 2:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 3:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q 
                if b == 5:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 6:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 7:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 9:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 10:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 11:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q
                if b == 12:
                    d_P[b,t] = buses[b].d_P
                    d_Q[b,t] = buses[b].d_Q

        load_power_forecast = {'active': d_P, 'reactive': d_Q}

        return load_power_forecast    # in pu

    def run_opf(self, control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, prior_reactive_restored_load, \
                mt_remaining_fuel, es_prior_soc, wind_power_forecast, solar_power_forecast):             
        """
        Run the Optimal Power Flow (OPF) 
        """

        #jl.include("lr_threephase_opf_mpc.jl")    # MPC/OPF model written in Julia/JuMP # Absolute path
        jl.include(os.path.dirname(self.cur_dir) + "/gym_opf_env/opf_envs/envs/lr_threephase_opf_mpc.jl")   # Relative path from the agent training scipt

        # MPC/OPF model inputs
        
        buses = self.network['buses']
        lines = self.network['lines']
        generators = self.network['generators']  # Non-renewable DERs
        windturbines = self.network['windturbines']
        pvs = self.network['pvs']
        storages = self.network['storages']

        mt_energy = mt_remaining_fuel
        es_soc = es_prior_soc  
     
        wt_power = wind_power_forecast
        pv_power = solar_power_forecast

        active_power_demanded = self.load_power_forecast['active']
        reactive_power_demanded = self.load_power_forecast['reactive']

        active_power_restored = prior_active_restored_load
        reactive_power_restored = prior_reactive_restored_load
        
        # Solve the OPF      
                                    
        results = jl.opf_mpc(buses, lines, generators, windturbines, pvs, storages, control_horizon, es_soc, mt_energy, \
                                wt_power, pv_power, active_power_demanded, reactive_power_demanded, \
                                    active_power_restored, reactive_power_restored, fuel_reserve, soc_reserve)
            
        opf_converged, objective_value, P_restored, Q_restored, Pmt, Qmt, Pwtb, Pwt_cut, Ppvs, Ppv_cut,\
            Pes, Qes, SOC_es, voltages, mu_P, mu_Q, frombus, tobus, P_lineflow, Q_lineflow = results                                   
     
        if opf_converged:
            # Exclude the non-load buses from the restored load arrays
            restored_kW = np.delete(P_restored, self.non_load_buses, 0)
            restored_kvar = np.delete(Q_restored, self.non_load_buses, 0)
        else:
            restored_kW = []
            Pmt = []
            Pes = []
            Pwtb = []
            Pwt_cut = []
            Ppvs = []
            Ppv_cut = [] 
            SOC_es = []
            restored_kvar = [] 
            Qmt = []
            Qes = [] 
            voltages = [] 

        return opf_converged, \
               restored_kW, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, \
               restored_kvar, Qmt, Qes, \
               SOC_es, voltages