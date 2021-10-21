import copy
import json
import os
import gym
from gym import spaces
import numpy as np
import pandas as pd

#from opf_envs.envs.lr_opf_jump import OPFJuMP      # OPF in Julia-JuMP
from opf_envs.envs.lr_opf_pyomo import OPFPyomo     # OPF in Python-Pyomo
from opf_envs.envs.lr_opf_pyomo import OPFPyomo2    
from opf_envs.envs.lr_opf_pyomo import OPFPyomo3    
from opf_envs.envs.renewable_power_forecast_reader import ForecastReader

FIVE_MIN_TO_HOUR = 5./60.
DEBUG = False
CONTROL_HORIZON_LEN = 72
STEPS_PER_HOUR = 12

REWARD_SCALING_FACTOR = 0.001
DEFAULT_ERROR_LEVEL = 0.1

CONTROL_HISTORY_DICT = {"active_load_status": [],
                        "reactive_load_status": [],
                        "pv_power": [],
                        "wt_power": [],
                        "mt_power": [],
                        "st_power": [],                        
                        "voltages": [],
                        "mt_remaining_fuel": [],
                        "st_soc": [],
                        "mt_fuel_reserve": [],
                        "st_soc_reserve": []}

CONTROL_HISTORY_DICT2 = {"active_load_status": [],
                        "reactive_load_status": [],
                        "pv_power": [],
                        "wt_power": [],
                        "mt_power": [],
                        "st_power": [],                        
                        "voltages": [],
                        "mt_remaining_fuel": [],
                        "st_soc": [],
                        "system_wide_reserve": []}

cur_dir = os.getcwd()
#PSEUDO_FORECASTS_DIR = cur_dir + "/data/exogenous_data/pfg_data/json_data"   # Absolute path
PSEUDO_FORECASTS_DIR = os.path.dirname(cur_dir) + "/gym_opf_env/opf_envs/envs/data/exogenous_data/pfg_data/json_data"  # Relative path from the agent training scipt
renewable_power_forecast_dir = os.path.dirname(cur_dir) + "/gym_opf_env/opf_envs/envs/data/exogenous_data" 

"""
Notations:
    ES - energy storage (battery)
    MT - microturbine
    OPF - optimal power flow
    MPC - model predictive control
    SOC - state of charge
"""

class ReservePolicyEnv(gym.Env):
    """ 
    gym environment for reserve policy learning problem.  

    """

    def __init__(self):    

        self.Zbase = 1
        self.Vbase = 4160
        self.Sbase = (self.Vbase**2)/self.Zbase
        self.w_to_kw_conv = 1000                
           
        self.wt_profile = None
        self.pv_profile = None
        self.renewable_pseudo_forecasts = None

        self.pv_max_gen = 300
        self.wind_max_gen = 400

        self.bus_name = ['650', '646', '645', '632', '633', '634', '611', '684', '671', '692', '675', '652', '680'] # IEEE 13bus balanced version
        self.num_of_bus = len(self.bus_name) 
        self.load_bus = [1, 2, 4, 5, 6, 8, 9, 10, 11]
        self.num_of_load = len(self.load_bus)  
        self.load_priority_weight = [1.0, 1.0, 0.9, 0.85, 0.8, 0.65, 0.45, 0.4, 0.3]

        self.psi = np.diag([100] * self.num_of_load)    # Penalty factor for shedding prior restored load
        self.alpha =  0.2                               # $/kWh -- penalty for wind power curtailment
        self.beta = 0.2                                 # $/kWh -- penalty for PV power curtailment

        self.es_soc_min = 20        # in %
        self.es_soc_max = 100       # in %

        #self.initial_restored_load = {'active': np.array([0] * self.num_of_load), 'reactive': np.array([0] * self.num_of_load)}    # in (kW, kvar)
        self.initial_active_restored_load = [0] * self.num_of_bus 
        self.initial_reactive_restored_load = [0] * self.num_of_bus 
        
        self.es_initial_soc = 100       # in %
        self.mt_initial_fuel = 1000     # in kWh

        #self.prior_restored_load = self.initial_restored_load
        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        self.opf_num_time_steps = CONTROL_HORIZON_LEN   # 1h long with 5 min resolution        
        self.simulation_step = 0
        self.done = False

        self.debug = None
        self.history = None
        self.enable_debug(DEBUG)

        self.time_of_day_list = None        

        self.error_level = DEFAULT_ERROR_LEVEL

        # Number of state variables
        self.num_state = 47         # 12 wind_forecast + 12 pv forecast + 9 active load + 9 reactive load + 5 scalar

        # Build action space        
        self.action_space = self.build_action_space()

        # Build observation space
        self.observation_space = self.build_observation_space()          
              
        # OPF model class object 
        #self.opf = OPFJuMP()        # OPF in Julia-JuMP
        self.opf = OPFPyomo()        # OPF in Python-Pyomo
           
        # Specify solver
        self.solver = 'glpk'   # can also be 'xpress_direct', 'cbc'     

    def enable_debug(self, debug_flag):
        self.debug = debug_flag
        if debug_flag:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)

    def reset(self, start_index=None, init_storage=None):
        
        self.simulation_step = 0  
        self.opf_num_time_steps = CONTROL_HORIZON_LEN           
        self.done = False
        
        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        # start_index = 180
        if start_index is None:
            # 0: Index for time 07/01 00:00
            # 8856: Index for time 07/31 18:00
            start_index = np.random.randint(0, 8856)

        print()
        print("Start Index for The Renewables:", start_index)
            
        self.time_of_day_list = []
        self.get_renewable_power_forecast(start_index)        

        state = self.get_state_forecast()

        return state

    def get_renewable_power_forecast(self, start_index):
        # Use future 1-hour forecasts as RL inputs.

        # Select which folder to read in pseudo forecasts.
        percentage_error_level = str(int(self.error_level * 100))         
        percentage_error_level = '10' if percentage_error_level == '0' else percentage_error_level

        with open(os.path.join(PSEUDO_FORECASTS_DIR,
                               percentage_error_level + 'p/forecasts_' + str(start_index) + '.json'), 'r') as fp:
            episode_renewable_data = json.load(fp)

        self.wt_profile = np.array(episode_renewable_data['actuals']['wind']) * self.wind_max_gen
        self.pv_profile = np.array(episode_renewable_data['actuals']['pv']) * self.pv_max_gen
        self.renewable_pseudo_forecasts = episode_renewable_data['forecasts']

        step_time = pd.to_datetime(episode_renewable_data['time'])
        for _ in range(CONTROL_HORIZON_LEN):
            self.time_of_day_list.append(step_time)
            step_time += pd.Timedelta('5M')   

    @staticmethod
    def get_trigonomical_representation(pd_datetime):

        daily_five_min_position = STEPS_PER_HOUR * pd_datetime.hour + pd_datetime.minute / 5
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        return np.sin(degree), np.cos(degree)

    def get_state_forecast(self):
        """ 
        Gather the system state at current step.

        - 13-Bus Balanced Case
        - State dimension is 47, these elements are:
            0-11: PV generation forecast for the next 6 hour
            12-23: Wind generation forecast for the next 1 hour
            24-32: Prior active restored loads (9 loads)
            33-41: Prior reactive restored loads (9 loads)
            42-46: [es_soc, mt_remaining_fuel, current_timestep, sinT, cosT] 
        """

        sim_step = self.simulation_step        
        forecast_horizon = STEPS_PER_HOUR 

        if self.error_level == 0.0:
            # Perfect forecasts
            pv_forecast = list(self.pv_profile[sim_step: sim_step + forecast_horizon]*self.w_to_kw_conv/self.Sbase)
            wt_forecast = list(self.wt_profile[sim_step: sim_step + forecast_horizon]*self.w_to_kw_conv/self.Sbase)
        else:
            pv_forecast = list(np.array(self.renewable_pseudo_forecasts[str(sim_step)]['pv'][:forecast_horizon])*self.pv_max_gen*self.w_to_kw_conv/self.Sbase)
            wt_forecast = list(np.array(self.renewable_pseudo_forecasts[str(sim_step)]['wind'][:forecast_horizon])*self.wind_max_gen*self.w_to_kw_conv/self.Sbase)

        current_active_load_status = []
        current_reactive_load_status = []
        for l in self.load_bus:
            current_active_load_status.append(self.prior_active_restored_load[l]*self.w_to_kw_conv/self.Sbase)
            current_reactive_load_status.append(self.prior_reactive_restored_load[l]*self.w_to_kw_conv/self.Sbase)

        for idx, p in enumerate(current_active_load_status):
            if p < 0 and abs(p) < 1e-4: # For numerical stability during RL's observation space boundary check
                current_active_load_status[idx] = 0.0 
 
        for idx, q in enumerate(current_reactive_load_status):
            if q < 0 and abs(q) < 1e-4: 
                current_reactive_load_status[idx] = 0.0  

        es_soc = self.es_current_storage/100.

        mt_fuel_remain = self.mt_remaining_fuel/self.mt_initial_fuel
        if mt_fuel_remain < 0 and abs(mt_fuel_remain) < 1e-4: 
            mt_fuel_remain = 0.0

        current_step = sim_step/CONTROL_HORIZON_LEN

        sin_t, cos_t = self.get_trigonomical_representation(self.time_of_day_list[sim_step])

        state = np.array(pv_forecast + wt_forecast + current_active_load_status + current_reactive_load_status + 
                        [es_soc, mt_fuel_remain, current_step, sin_t, cos_t])

        return state

    def step(self, action):
        """ Implementing one step of control.

        This function consists of 3 parts:
          1. Action pre-processing: convert the normalized control to their original range.        
          2. Control implementation: a. run the OPF; b. update the ES and MT status.
          3. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (2,). 
            - The second element is the MT end-of-horizon fuel reserve: (-1,1) --> (0, self.mt_remaining_fuel).
            - The first element is the ES end-of-horizon SOC reserve: (-1,1) --> (20%, self.es_current_storage).
        """

        # Step 1: Pre-process actions
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map the RL agent chosen actions, at this step, to the MT fuel (in kWh) and ES SOC (in %) reserve requirements.

        fuel_reserve = self.mt_remaining_fuel*(action[0]+1)/2                                           # (-1, 1) -> (0, self.mt_remaining_fuel) MT fuel reserve
        soc_reserve = (self.es_current_storage - self.es_soc_min)*(action[1]+1)/2 + self.es_soc_min     # (-1, 1) -> (self.es_soc_min, self.es_current_storage) ES SOC reserve

        print()
        print("Simulation Step:", self.simulation_step)

        print()
        print("MT Remaining Fuel:", self.mt_remaining_fuel)
        print("ES SOC Level:", self.es_current_storage)
        print("Agent Chosen Action:", action)
        print("Fuel Reserve Mapped:", fuel_reserve)
        print("SOC Reserve Mapped:", soc_reserve)       

        # Renewable power
        if self.error_level == 0.0:
            # Perfect forecasts
            p_pv = list(self.pv_profile[self.simulation_step: self.simulation_step + self.opf_num_time_steps]*self.w_to_kw_conv/self.Sbase)[:self.opf_num_time_steps]
            p_wt = list(self.wt_profile[self.simulation_step: self.simulation_step + self.opf_num_time_steps]*self.w_to_kw_conv/self.Sbase)[:self.opf_num_time_steps]
        else:
            p_pv = list(np.array(self.renewable_pseudo_forecasts[str(self.simulation_step)]['pv'][:self.opf_num_time_steps])*self.pv_max_gen*self.w_to_kw_conv/self.Sbase)
            p_wt = list(np.array(self.renewable_pseudo_forecasts[str(self.simulation_step)]['wind'][:self.opf_num_time_steps])*self.wind_max_gen*self.w_to_kw_conv/self.Sbase)
        
        # Step 2: Control implementation - Compute OPF and Update the MT fuel and ES SOC levels
       
        opf_solution = self.get_opf_solution(self.opf_num_time_steps, fuel_reserve, soc_reserve, self.prior_active_restored_load, \
                                             self.prior_reactive_restored_load, self.mt_remaining_fuel, \
                                             self.es_current_storage, p_wt, p_pv)     

        opf_converged, P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, Q_restored, Qmt, Qwtb, Qpvs, Qes, SOC_es, voltages = opf_solution
      
        if opf_converged:  
           
            # Update the MT fuel and ES SOC levels

            self.mt_remaining_fuel -= Pmt[0] * FIVE_MIN_TO_HOUR
            self.es_current_storage = SOC_es[0]

            if self.mt_remaining_fuel < 0 and abs(self.mt_remaining_fuel) < 1e-4:  # For numerical stability during Pyomo's boundary check
                self.mt_remaining_fuel = 0.0

            # Step 3: Post-control process

            # Calculate reward:
                # Consists of three parts: load restoration reward, load shedding penalty and renewable power curtailment penalty

            # 1. Load restoration reward

            active_load_restored = P_restored[:,0]      # the restored load at the current step
            reactive_load_restored = Q_restored[:,0]    # the restored load at the current step

            load_restoration_reward = np.dot(self.load_priority_weight, active_load_restored) + \
                                        np.dot(self.load_priority_weight, reactive_load_restored)
                    
            # 2. Load shedding penalty

            # Prior restored loads - re-organize by excluding non-load buses
            prior_active_restored_load_with_only_load_bus = []
            prior_reactive_restored_load_with_only_load_bus = []
            for l in self.load_bus:
                prior_active_restored_load_with_only_load_bus.append(self.prior_active_restored_load[l])
                prior_reactive_restored_load_with_only_load_bus.append(self.prior_reactive_restored_load[l])
            
            active_load_shedded = [max(0.0, (prior_active_restored_load_with_only_load_bus[idx] - active_load_restored[idx])) for idx in range(self.num_of_load)]
            reactive_load_shedded = [max(0.0, (prior_reactive_restored_load_with_only_load_bus[idx] - reactive_load_restored[idx])) for idx in range(self.num_of_load)]
            
            print()        
            print("Active Shedded Loads:", active_load_shedded)
            print("Reactive Shedded Loads:", reactive_load_shedded)           

            active_load_shedded_by_psi = np.matmul(self.psi, np.array(active_load_shedded).reshape([-1, 1]))
            reactive_load_shedded_by_psi = np.matmul(self.psi, np.array(reactive_load_shedded).reshape([-1, 1]))

            load_shedding_penalty = float(np.dot(self.load_priority_weight, active_load_shedded_by_psi)) + \
                                        float(np.dot(self.load_priority_weight, reactive_load_shedded_by_psi))

            # 3. Renewable power curtailment penalty

            curtailed_wind_power = Pwt_cut[0]
            curtailed_solar_power = Ppv_cut[0]

            renewable_power_curtailment_penalty = self.alpha*curtailed_wind_power + self.beta*curtailed_solar_power

            # Update the prior restored load (for next control step)

            for idx, b in enumerate(self.load_bus):
                self.prior_active_restored_load[b] = active_load_restored[idx]
                self.prior_reactive_restored_load[b] = reactive_load_restored[idx]

            state = self.get_state_forecast()

            reward = (load_restoration_reward - load_shedding_penalty - renewable_power_curtailment_penalty) * REWARD_SCALING_FACTOR  

            print()        
            print("Reward:", reward)

            if reward < 0:
                print()
                print("Reward is -ve")
                print("Agent Chosen Action:", action)
                print("Fuel Reserve Mapped:", fuel_reserve)
                print("SOC Reserve Mapped:", soc_reserve) 

        else:
            # Terminal state has been reached if no solution to the opf is found.

            print()
            print("OPF Infeasibility Happened")
            print("Agent Chosen MT Fuel Reserve: ", fuel_reserve)
            print("Agent Chosen ST SOC Reserve: ", soc_reserve)                
         
            state = self.terminal_state(self.num_state)
            reward = -1000000           # Very large reward

            Ppvs = state[:self.opf_num_time_steps]
            Pwtb = state[self.opf_num_time_steps: 2*self.opf_num_time_steps]  
            prior_active_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps:2*self.opf_num_time_steps+self.num_of_load]
            prior_reactive_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps+self.num_of_load: 2*self.opf_num_time_steps+2*self.num_of_load]
            Pmt = np.array([0] * self.opf_num_time_steps)
            Qmt = np.array([0] * self.opf_num_time_steps)
            Pes = np.array([0] * self.opf_num_time_steps)
            Qes = np.array([0] * self.opf_num_time_steps)
            voltages = np.zeros((self.num_of_bus, self.opf_num_time_steps))

            # TODO: Use previous states ??     
            
        # Update the timestep

        self.simulation_step += 1

        # Update the forecast/control horizon

        self.opf_num_time_steps -= 1
    
        # Terminate the episode if the final control step is reached

        self.done = True if self.simulation_step >= CONTROL_HORIZON_LEN else False
        
        if self.debug:
            self.history['active_load_status'].append(active_load_restored)
            self.history['reactive_load_status'].append(reactive_load_restored)
            self.history['pv_power'].append(Ppvs[0])
            self.history['wt_power'].append(Pwtb[0])
            self.history['mt_power'].append([Pmt[0], Qmt[0]])
            self.history['st_power'].append([Pes[0], Qes[0]])
            self.history['voltages'].append(voltages[:,0])
            self.history['mt_remaining_fuel'].append(self.mt_remaining_fuel)
            self.history['st_soc'].append(self.es_current_storage) 
            self.history['mt_fuel_reserve'].append(fuel_reserve)
            self.history['st_soc_reserve'].append(soc_reserve)
            self.history['voltage_bus_names'] = self.bus_name        

        return state, reward, self.done, {}

    def get_opf_solution(self, control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                         prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                         wind_power_forecast, solar_power_forecast):

        # Call the Julia/JuMP OPF/MPC
        
        #opf_solution = self.opf.run_opf(control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
        #                                prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
        #                                wind_power_forecast, solar_power_forecast)

        # Call the Python/Pyomo OPF/MPC
  
        # TODO: For the purpose of performance improvement update the pyomo model parameters instead of rebuilding the model at each simulation step.

        model = self.opf.build_load_restoration_model(control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                                                 prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                                 wind_power_forecast, solar_power_forecast)

        opf_solution = self.opf.compute_solution(model, self.solver, solver_tee=False, solver_options=None)

        return opf_solution
    
    def build_action_space(self):
        """
        Build the action space available to the agent to choose from.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        lower_bounds = np.array([-1.0] * 2)
        upper_bounds = np.array([1.0] * 2)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float32)

        return space

    def build_observation_space(self):
        """
        Build the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space of the environment.
        """

        dim_obs = self.num_state

        lower_bounds = np.array([0.0] * (dim_obs - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 
        upper_bounds = np.array([1.0] * dim_obs)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float)

        return space   

    def update_observation_space(self):
        """
        Update the observation space, as the renewable power forecast horizon changes.

        Returns
        -------
        gym.spaces.Box
            The updated observation space of the environment.
        """

        dim_obs = 2*self.opf_num_time_steps + 18 + 5

        lower_bounds = np.array([0.0] * (dim_obs - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 
        upper_bounds = np.array([1.0] * dim_obs)

        self.observation_space = spaces.Box(low=lower_bounds, 
                                            high=upper_bounds, 
                                            dtype=np.float)
    
    def terminal_state(self, num_state):
        """
        Return a vector of lower of bounds of the observation space (arbitrarily chosen as terminal state).

        Parameters
        ----------
        num_state : int
            The length of the vector to return.

        Returns
        -------
        numpy.ndarray
            A vector of size num_state whose values are lower bounds of the observation space.
        """

        state = np.array([0.0] * (num_state - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 

        return state

    def get_control_history(self):

        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),            
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']), 
            'mt_fuel_reserve': np.array(self.history['mt_fuel_reserve']),
            'st_soc_reserve': np.array(self.history['st_soc_reserve']),           
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'active_load_status': np.array(self.history['active_load_status']),
            'reactive_load_status': np.array(self.history['reactive_load_status']),
            'time_stamps': self.time_of_day_list
            }

        return results


##############################################################################################################
########################################### ENV #2 ###########################################################
##############################################################################################################


class ReservePolicyEnv2(gym.Env):
    """ 
    gym environment for reserve policy learning problem.

    compared to the previous env, this env has one action which is the system-wide end-of-horizon reserve requirement.  

    """

    def __init__(self):    

        self.Zbase = 1
        self.Vbase = 4160
        self.Sbase = (self.Vbase**2)/self.Zbase
        self.Cbase = 800
        self.w_to_kw_conv = 1000
                   
        self.wt_profile = None
        self.pv_profile = None
        self.renewable_pseudo_forecasts = None

        self.pv_max_gen = 300
        self.wind_max_gen = 400

        self.bus_name = ['650', '646', '645', '632', '633', '634', '611', '684', '671', '692', '675', '652', '680'] 
        self.num_of_bus = len(self.bus_name) 
        self.load_bus = [1, 2, 4, 5, 6, 8, 9, 10, 11]
        self.num_of_load = len(self.load_bus)  
        self.load_priority_weight = [1.0, 1.0, 0.9, 0.85, 0.8, 0.65, 0.45, 0.4, 0.3]

        self.psi = np.diag([100] * self.num_of_load)    
        self.alpha =  0.2                               
        self.beta = 0.2                                

        self.es_cap = self.Cbase
        self.es_soc_min = 20        
        self.es_soc_max = 100       

        self.initial_active_restored_load = [0] * self.num_of_bus 
        self.initial_reactive_restored_load = [0] * self.num_of_bus 
        
        self.es_initial_soc = 100      
        self.mt_initial_fuel = 1000    
        
        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        self.opf_num_time_steps = STEPS_PER_HOUR         
        self.simulation_step = 0
        self.done = False

        self.debug = None
        self.history = None
        self.enable_debug(DEBUG)

        self.time_of_day_list = None        

        self.error_level = DEFAULT_ERROR_LEVEL

        # Number of state variables
        self.num_state = 47        

        # Build action space        
        self.action_space = self.build_action_space()

        # Build observation space
        self.observation_space = self.build_observation_space()          
              
        # OPF model class object         
        self.opf = OPFPyomo2()        
   
        # Specify solver
        self.solver = 'glpk'      

    def enable_debug(self, debug_flag):
        self.debug = debug_flag
        if debug_flag:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT2)

    def reset(self, start_index=None, init_storage=None):
        
        self.simulation_step = 0              
        self.done = False        

        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc
      
        if start_index is None:
            start_index = np.random.randint(0, 8856)
            
        self.time_of_day_list = []
        self.get_renewable_power_forecast(start_index)        

        state = self.get_state_forecast()

        return state

    def get_renewable_power_forecast(self, start_index):
       
        percentage_error_level = str(int(self.error_level * 100))         
        percentage_error_level = '10' if percentage_error_level == '0' else percentage_error_level

        with open(os.path.join(PSEUDO_FORECASTS_DIR,
                               percentage_error_level + 'p/forecasts_' + str(start_index) + '.json'), 'r') as fp:
            episode_renewable_data = json.load(fp)

        self.wt_profile = np.array(episode_renewable_data['actuals']['wind']) * self.wind_max_gen
        self.pv_profile = np.array(episode_renewable_data['actuals']['pv']) * self.pv_max_gen
        self.renewable_pseudo_forecasts = episode_renewable_data['forecasts']

        step_time = pd.to_datetime(episode_renewable_data['time'])
        for _ in range(CONTROL_HORIZON_LEN):
            self.time_of_day_list.append(step_time)
            step_time += pd.Timedelta('5M')

    @staticmethod
    def get_trigonomical_representation(pd_datetime):

        daily_five_min_position = STEPS_PER_HOUR * pd_datetime.hour + pd_datetime.minute / 5
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        return np.sin(degree), np.cos(degree)

    def get_state_forecast(self):
        """ 
        Gather the system state at current step.

        - 13-Bus Balanced Case
        - State dimension is 47, these elements are:
            0-11: PV generation forecast for the next 1 hour
            12-23: Wind generation forecast for the next 1 hour
            24-32: Prior active restored loads (9 loads)
            33-41: Prior reactive restored loads (9 loads)
            42-46: [es_soc, mt_remaining_fuel, current_timestep, sinT, cosT] 
        """

        sim_step = self.simulation_step        
        forecast_horizon = STEPS_PER_HOUR

        if self.error_level == 0.0:           
            pv_forecast = list(self.pv_profile[sim_step: sim_step + forecast_horizon]*self.w_to_kw_conv/self.Sbase)
            wt_forecast = list(self.wt_profile[sim_step: sim_step + forecast_horizon]*self.w_to_kw_conv/self.Sbase)
        else:
            pv_forecast = list(np.array(self.renewable_pseudo_forecasts[str(sim_step)]['pv'][:forecast_horizon])*self.pv_max_gen*self.w_to_kw_conv/self.Sbase)
            wt_forecast = list(np.array(self.renewable_pseudo_forecasts[str(sim_step)]['wind'][:forecast_horizon])*self.wind_max_gen*self.w_to_kw_conv/self.Sbase)

        current_active_load_status = []
        current_reactive_load_status = []
        for l in self.load_bus:
            current_active_load_status.append(self.prior_active_restored_load[l]*self.w_to_kw_conv/self.Sbase)
            current_reactive_load_status.append(self.prior_reactive_restored_load[l]*self.w_to_kw_conv/self.Sbase)

        for idx, p in enumerate(current_active_load_status):
            if p < 0 and abs(p) < 1e-4: 
                current_active_load_status[idx] = 0.0 
 
        for idx, q in enumerate(current_reactive_load_status):
            if q < 0 and abs(q) < 1e-4: 
                current_reactive_load_status[idx] = 0.0  

        es_soc = self.es_current_storage/100.

        mt_fuel_remain = self.mt_remaining_fuel/self.mt_initial_fuel
        if mt_fuel_remain < 0 and abs(mt_fuel_remain) < 1e-4: 
            mt_fuel_remain = 0.0

        current_step = sim_step/CONTROL_HORIZON_LEN

        sin_t, cos_t = self.get_trigonomical_representation(self.time_of_day_list[sim_step])

        state = np.array(pv_forecast + wt_forecast + current_active_load_status + current_reactive_load_status + 
                        [es_soc, mt_fuel_remain, current_step, sin_t, cos_t])

        return state

    def step(self, action):
        """ Implementing one step of control.

        This function consists of 3 parts:
          1. Action pre-processing: convert the normalized control to their original range.        
          2. Control implementation: a. run the OPF; b. update the ES and MT status.
          3. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (1,). 
            - The system-wide end-of-horizon reserve requirement.
            - Takes values in the ranage: (-1,1) --> (0, self.mt_remaining_fuel + (self.es_current_storage - self.es_soc_min)*self.es_cap/100) 
        """        

        # Step 1: Pre-process actions        

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map the RL agent chosen action, at this step, to the system-wide reserve requirement (in kWh).
        # (-1,1) --> (0, self.mt_remaining_fuel + (self.es_current_storage - self.es_soc_min)*self.es_cap/100)

        system_wide_reserve = (self.mt_remaining_fuel + (self.es_current_storage - self.es_soc_min)*self.es_cap/100)*(action[0]+1)/2

        if system_wide_reserve < 0 and abs(system_wide_reserve) < 1e-4:
            system_wide_reserve = 0.0 
     
        print()
        print("Simulation Step:", self.simulation_step)

        print()
        print("MT Remaining Fuel:", self.mt_remaining_fuel)
        print("ES SOC Level:", self.es_current_storage)
        print("Agent Chosen Action:", action)
        print("System-wide Reserve Mapped:", system_wide_reserve)      

        # Renewable power
        if self.error_level == 0.0:
            p_pv = list(self.pv_profile[self.simulation_step: self.simulation_step + self.opf_num_time_steps]*self.w_to_kw_conv/self.Sbase)[:self.opf_num_time_steps]
            p_wt = list(self.wt_profile[self.simulation_step: self.simulation_step + self.opf_num_time_steps]*self.w_to_kw_conv/self.Sbase)[:self.opf_num_time_steps]
        else:
            p_pv = list(np.array(self.renewable_pseudo_forecasts[str(self.simulation_step)]['pv'][:self.opf_num_time_steps])*self.pv_max_gen*self.w_to_kw_conv/self.Sbase)
            p_wt = list(np.array(self.renewable_pseudo_forecasts[str(self.simulation_step)]['wind'][:self.opf_num_time_steps])*self.wind_max_gen*self.w_to_kw_conv/self.Sbase)
        
        # Step 2: Control implementation - Compute OPF and Update the MT fuel and ES SOC levels
       
        opf_solution = self.get_opf_solution(self.opf_num_time_steps, system_wide_reserve, self.prior_active_restored_load, \
                                             self.prior_reactive_restored_load, self.mt_remaining_fuel, self.es_current_storage, \
                                             p_wt, p_pv)     

        opf_converged, P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, Q_restored, Qmt, Qwtb, Qpvs, Qes, SOC_es, voltages = opf_solution
      
        if opf_converged:  
           
            # Update the MT fuel and ES SOC levels

            self.mt_remaining_fuel -= Pmt[0] * FIVE_MIN_TO_HOUR
            self.es_current_storage = SOC_es[0]

            if self.mt_remaining_fuel < 0 and abs(self.mt_remaining_fuel) < 1e-4:  
                self.mt_remaining_fuel = 0.0

            # Step 3: Post-control process

            # Calculate reward:
                # Consists of three parts: load restoration reward, load shedding penalty and renewable power curtailment penalty

            # 1. Load restoration reward

            active_load_restored = P_restored[:,0]      
            reactive_load_restored = Q_restored[:,0]   

            load_restoration_reward = np.dot(self.load_priority_weight, active_load_restored) + \
                                        np.dot(self.load_priority_weight, reactive_load_restored)
                    
            # 2. Load shedding penalty

            # Prior restored loads - re-organize by excluding non-load buses
            prior_active_restored_load_with_only_load_bus = []
            prior_reactive_restored_load_with_only_load_bus = []
            for l in self.load_bus:
                prior_active_restored_load_with_only_load_bus.append(self.prior_active_restored_load[l])
                prior_reactive_restored_load_with_only_load_bus.append(self.prior_reactive_restored_load[l])
            
            active_load_shedded = [max(0.0, (prior_active_restored_load_with_only_load_bus[idx] - active_load_restored[idx])) for idx in range(self.num_of_load)]
            reactive_load_shedded = [max(0.0, (prior_reactive_restored_load_with_only_load_bus[idx] - reactive_load_restored[idx])) for idx in range(self.num_of_load)]
            
            print()        
            print("Active Shedded Loads:", active_load_shedded)
            print("Reactive Shedded Loads:", reactive_load_shedded)           

            active_load_shedded_by_psi = np.matmul(self.psi, np.array(active_load_shedded).reshape([-1, 1]))
            reactive_load_shedded_by_psi = np.matmul(self.psi, np.array(reactive_load_shedded).reshape([-1, 1]))

            load_shedding_penalty = float(np.dot(self.load_priority_weight, active_load_shedded_by_psi)) + \
                                        float(np.dot(self.load_priority_weight, reactive_load_shedded_by_psi))

            # 3. Renewable power curtailment penalty

            curtailed_wind_power = Pwt_cut[0]
            curtailed_solar_power = Ppv_cut[0]

            renewable_power_curtailment_penalty = self.alpha*curtailed_wind_power + self.beta*curtailed_solar_power

            # Update the prior restored load (for next control step)

            for idx, b in enumerate(self.load_bus):
                self.prior_active_restored_load[b] = active_load_restored[idx]
                self.prior_reactive_restored_load[b] = reactive_load_restored[idx]

            state = self.get_state_forecast()

            reward = (load_restoration_reward - load_shedding_penalty - renewable_power_curtailment_penalty) * REWARD_SCALING_FACTOR  

            if reward < 0:
                print()
                print("Reward is -ve")
                print("Agent Chosen Action:", action)
                print("System-wide Reserve Mapped:", system_wide_reserve)                

        else:
            # Terminal state has been reached if no solution to the opf is found.

            print()
            print("OPF Infeasibility Happened")
            print("Agent Chosen System-wide Reserve: ", system_wide_reserve)                      
         
            state = self.terminal_state(self.num_state)
            reward = -1000000           # Very large reward

            Ppvs = state[:self.opf_num_time_steps]
            Pwtb = state[self.opf_num_time_steps: 2*self.opf_num_time_steps]  
            prior_active_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps:2*self.opf_num_time_steps+self.num_of_load]
            prior_reactive_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps+self.num_of_load: 2*self.opf_num_time_steps+2*self.num_of_load]
            Pmt = np.array([0] * self.opf_num_time_steps)
            Qmt = np.array([0] * self.opf_num_time_steps)
            Pes = np.array([0] * self.opf_num_time_steps)
            Qes = np.array([0] * self.opf_num_time_steps)
            voltages = np.zeros((self.num_of_bus, self.opf_num_time_steps))

            # TODO: Use previous states ??     
            
        # Update the timestep

        self.simulation_step += 1
            
        # Terminate the episode if the final control step is reached

        self.done = True if self.simulation_step >= CONTROL_HORIZON_LEN else False
        
        if self.debug:
            self.history['active_load_status'].append(active_load_restored)
            self.history['reactive_load_status'].append(reactive_load_restored)
            self.history['pv_power'].append(Ppvs[0])
            self.history['wt_power'].append(Pwtb[0])
            self.history['mt_power'].append([Pmt[0], Qmt[0]])
            self.history['st_power'].append([Pes[0], Qes[0]])
            self.history['voltages'].append(voltages[:,0])
            self.history['mt_remaining_fuel'].append(self.mt_remaining_fuel)
            self.history['st_soc'].append(self.es_current_storage) 
            self.history['system_wide_reserve'].append(system_wide_reserve)            
            self.history['voltage_bus_names'] = self.bus_name        

        return state, reward, self.done, {}

    def get_opf_solution(self, control_horizon, system_wide_reserve, prior_active_restored_load, \
                         prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                         wind_power_forecast, solar_power_forecast):
       
        # Build the Python/Pyomo OPF/MPC-based lr model and compute the model solution
  
        model = self.opf.build_load_restoration_model(control_horizon, system_wide_reserve, prior_active_restored_load, \
                                                      prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                                      wind_power_forecast, solar_power_forecast)

        opf_solution = self.opf.compute_solution(model, self.solver, solver_tee=False, solver_options=None)

        return opf_solution
    
    def build_action_space(self):
        """
        Build the action space available to the agent to choose from.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        lower_bounds = np.array([-1.0])
        upper_bounds = np.array([1.0])

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float32)

        return space

    def build_observation_space(self):
        """
        Build the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space of the environment.
        """

        dim_obs = self.num_state

        lower_bounds = np.array([0.0] * (dim_obs - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 
        upper_bounds = np.array([1.0] * dim_obs)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float)

        return space   
    
    def terminal_state(self, num_state):
        """
        Return a vector of lower of bounds of the observation space (arbitrarily chosen as terminal state).

        Parameters
        ----------
        num_state : int
            The length of the vector to return.

        Returns
        -------
        numpy.ndarray
            A vector of size num_state whose values are lower bounds of the observation space.
        """

        state = np.array([0.0] * (num_state - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 

        return state

    def get_control_history(self):

        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),            
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']), 
            'system_wide_reserve': np.array(self.history['system_wide_reserve']),                     
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'active_load_status': np.array(self.history['active_load_status']),
            'reactive_load_status': np.array(self.history['reactive_load_status']),
            'time_stamps': self.time_of_day_list
            }

        return results


##############################################################################################################
########################################### ENV #3 ###########################################################
##############################################################################################################


class ReservePolicyEnv3(gym.Env):
    """ 
    gym environment for reserve policy learning problem.

    this env is similar to the 1st env except it considers 6h long renewable power forecasts

    """  

    def __init__(self):    

        self.Zbase = 1
        self.Vbase = 4160
        self.Sbase = (self.Vbase**2)/self.Zbase
        self.w_to_kw_conv = 1000        
        
        self.wind_power_forecasts = None
        self.pv_power_forecasts = None           
        
        self.pv_max_gen = 300
        self.wind_max_gen = 400

        self.bus_name = ['650', '646', '645', '632', '633', '634', '611', '684', '671', '692', '675', '652', '680'] # IEEE 13bus balanced version
        self.num_of_bus = len(self.bus_name) 
        self.load_bus = [1, 2, 4, 5, 6, 8, 9, 10, 11]
        self.num_of_load = len(self.load_bus)  
        self.load_priority_weight = [1.0, 1.0, 0.9, 0.85, 0.8, 0.65, 0.45, 0.4, 0.3]

        self.psi = np.diag([100] * self.num_of_load)    # Penalty factor for shedding prior restored load
        self.alpha =  0.2                               # $/kWh -- penalty for wind power curtailment
        self.beta = 0.2                                 # $/kWh -- penalty for PV power curtailment

        self.es_soc_min = 20        # in %
        self.es_soc_max = 100       # in %

        self.initial_active_restored_load = [0] * self.num_of_bus 
        self.initial_reactive_restored_load = [0] * self.num_of_bus 
        
        self.es_initial_soc = 100       # in %
        self.mt_initial_fuel = 1000     # in kWh

        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        self.opf_num_time_steps = CONTROL_HORIZON_LEN    # 6h long with 5 min resolution        
        self.simulation_step = 0
        self.done = False

        self.debug = None
        self.history = None
        self.enable_debug(DEBUG)

        self.time_of_day_list = None        
        
        # Number of state variables
        self.num_state = 167         # 72 wind_forecast + 72 pv forecast + 9 active load + 9 reactive load + 5 scalar

        # Build action space        
        self.action_space = self.build_action_space()

        # Build observation space
        self.observation_space = self.build_observation_space()          
              
        # OPF model class object 
        #self.opf = OPFJuMP()        # OPF in Julia-JuMP
        self.opf = OPFPyomo()        # OPF in Python-Pyomo
           
        # Specify solver
        self.solver = 'glpk'   # can also be 'xpress_direct', 'cbc'     

    def enable_debug(self, debug_flag):
        self.debug = debug_flag
        if debug_flag:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)

    def reset(self, start_index=None, init_storage=None):
        
        self.simulation_step = 0             
        self.done = False
        
        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        # start_index = 180
        if start_index is None:
            # 0: Index for time 07/01 00:00
            # 8856: Index for time 07/31 18:00
            start_index = np.random.randint(0, 8856)

        print()
        print("Start Index for The Renewables:", start_index)
            
        self.time_of_day_list = []        
        self.get_renewable_power_forecast_6h(start_index)

        state = self.get_state_forecast()

        return state
    
    def get_renewable_power_forecast_6h(self, start_index):        
        # Use future 6-hour forecasts as RL inputs.        
              
        os.chdir(renewable_power_forecast_dir)  
                
        forecast_reader = ForecastReader()
        available_datetimes = forecast_reader.available_dates()

        forecast_horizon = CONTROL_HORIZON_LEN 

        wind_power_forecasts = np.empty((0, forecast_horizon))
        pv_power_forecasts = np.empty((0, forecast_horizon))                        

        available_datetimes = list(available_datetimes)
        available_datetimes_for_training = available_datetimes[:8928]   # 07/01 00:00 - 07/31 23:55
        
        for datetime in available_datetimes_for_training[start_index: start_index+forecast_horizon]:
        
            renewable_power_forecast = forecast_reader.get_forecast(datetime)

            wind_power_forecast = (np.array(renewable_power_forecast['wind_gen'])*self.wind_max_gen)[:forecast_horizon]
            wind_power_forecasts = np.append(wind_power_forecasts, [wind_power_forecast], axis=0)     # in kW
            self.wind_power_forecasts = wind_power_forecasts

            pv_power_forecast = (np.array(renewable_power_forecast['pv_gen'])*self.pv_max_gen)[:forecast_horizon]
            pv_power_forecasts = np.append(pv_power_forecasts, [pv_power_forecast], axis=0)   # in kW
            self.pv_power_forecasts = pv_power_forecasts 
    
        step_time = available_datetimes_for_training[start_index]
        for _ in range(CONTROL_HORIZON_LEN):
            self.time_of_day_list.append(step_time)
            step_time += pd.Timedelta('5M')

        os.chdir(cur_dir)         

    @staticmethod
    def get_trigonomical_representation(pd_datetime):

        daily_five_min_position = STEPS_PER_HOUR * pd_datetime.hour + pd_datetime.minute / 5
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        return np.sin(degree), np.cos(degree)

    def get_state_forecast(self):
        """ 
        Gather the system state at current step.

        - 13-Bus Balanced Case
        - State dimension is 167, these elements are:
            0-71: PV generation forecast for the next 6 hour
            72-143: Wind generation forecast for the next 1 hour
            144-152: Prior active restored loads (9 loads)
            153-161: Prior reactive restored loads (9 loads)
            162-166: [es_soc, mt_remaining_fuel, current_timestep, sinT, cosT] 
        """

        sim_step = self.simulation_step      
        
        pv_forecast = list(self.pv_power_forecasts[sim_step,:]*self.w_to_kw_conv/self.Sbase)
        wt_forecast = list(self.wind_power_forecasts[sim_step,:]*self.w_to_kw_conv/self.Sbase)

        current_active_load_status = []
        current_reactive_load_status = []
        for l in self.load_bus:
            current_active_load_status.append(self.prior_active_restored_load[l]*self.w_to_kw_conv/self.Sbase)
            current_reactive_load_status.append(self.prior_reactive_restored_load[l]*self.w_to_kw_conv/self.Sbase)

        for idx, p in enumerate(current_active_load_status):
            if p < 0 and abs(p) < 1e-4: # For numerical stability during RL's observation space boundary check
                current_active_load_status[idx] = 0.0 
 
        for idx, q in enumerate(current_reactive_load_status):
            if q < 0 and abs(q) < 1e-4: 
                current_reactive_load_status[idx] = 0.0  

        es_soc = self.es_current_storage/100.

        mt_fuel_remain = self.mt_remaining_fuel/self.mt_initial_fuel
        if mt_fuel_remain < 0 and abs(mt_fuel_remain) < 1e-4: 
            mt_fuel_remain = 0.0

        current_step = sim_step/CONTROL_HORIZON_LEN

        sin_t, cos_t = self.get_trigonomical_representation(self.time_of_day_list[sim_step])

        state = np.array(pv_forecast + wt_forecast + current_active_load_status + current_reactive_load_status + 
                        [es_soc, mt_fuel_remain, current_step, sin_t, cos_t])

        return state

    def step(self, action):
        """ Implementing one step of control.

        This function consists of 3 parts:
          1. Action pre-processing: convert the normalized control to their original range.        
          2. Control implementation: a. run the OPF; b. update the ES and MT status.
          3. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (2,). 
            - The second element is the MT end-of-horizon fuel reserve: (-1,1) --> (0, self.mt_remaining_fuel).
            - The first element is the ES end-of-horizon SOC reserve: (-1,1) --> (20%, self.es_current_storage).
        """

        # Step 1: Pre-process actions
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map the RL agent chosen actions, at this step, to the MT fuel (in kWh) and ES SOC (in %) reserve requirements.

        fuel_reserve = self.mt_remaining_fuel*(action[0]+1)/2                                           # (-1, 1) -> (0, self.mt_remaining_fuel) MT fuel reserve
        soc_reserve = (self.es_current_storage - self.es_soc_min)*(action[1]+1)/2 + self.es_soc_min     # (-1, 1) -> (self.es_soc_min, self.es_current_storage) ES SOC reserve

        print()
        print("Simulation Step:", self.simulation_step)

        print()
        print("MT Remaining Fuel:", self.mt_remaining_fuel)
        print("ES SOC Level:", self.es_current_storage)
        print("Agent Chosen Action:", action)
        print("Fuel Reserve Mapped:", fuel_reserve)
        print("SOC Reserve Mapped:", soc_reserve) 

        # Renewable power
        p_pv= list(self.pv_power_forecasts[self.simulation_step,:]*self.w_to_kw_conv/self.Sbase)
        p_wt = list(self.wind_power_forecasts[self.simulation_step,:]*self.w_to_kw_conv/self.Sbase)      
        
        # Step 2: Control implementation - Compute OPF and Update the MT fuel and ES SOC levels
       
        opf_solution = self.get_opf_solution(self.opf_num_time_steps, fuel_reserve, soc_reserve, self.prior_active_restored_load, \
                                             self.prior_reactive_restored_load, self.mt_remaining_fuel, \
                                             self.es_current_storage, p_wt, p_pv)     

        opf_converged, P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, Q_restored, Qmt, Qwtb, Qpvs, Qes, SOC_es, voltages = opf_solution
      
        if opf_converged:  
           
            # Update the MT fuel and ES SOC levels

            self.mt_remaining_fuel -= Pmt[0] * FIVE_MIN_TO_HOUR
            self.es_current_storage = SOC_es[0]

            if self.mt_remaining_fuel < 0 and abs(self.mt_remaining_fuel) < 1e-4:  # For numerical stability during Pyomo's boundary check
                self.mt_remaining_fuel = 0.0

            # Step 3: Post-control process

            # Calculate reward:
                # Consists of three parts: load restoration reward, load shedding penalty and renewable power curtailment penalty

            # 1. Load restoration reward

            active_load_restored = P_restored[:,0]      # the restored load at the current step
            reactive_load_restored = Q_restored[:,0]    # the restored load at the current step

            load_restoration_reward = np.dot(self.load_priority_weight, active_load_restored) + \
                                        np.dot(self.load_priority_weight, reactive_load_restored)
                    
            # 2. Load shedding penalty

            # Prior restored loads - re-organize by excluding non-load buses
            prior_active_restored_load_with_only_load_bus = []
            prior_reactive_restored_load_with_only_load_bus = []
            for l in self.load_bus:
                prior_active_restored_load_with_only_load_bus.append(self.prior_active_restored_load[l])
                prior_reactive_restored_load_with_only_load_bus.append(self.prior_reactive_restored_load[l])
            
            active_load_shedded = [max(0.0, (prior_active_restored_load_with_only_load_bus[idx] - active_load_restored[idx])) for idx in range(self.num_of_load)]
            reactive_load_shedded = [max(0.0, (prior_reactive_restored_load_with_only_load_bus[idx] - reactive_load_restored[idx])) for idx in range(self.num_of_load)]
            
            print()        
            print("Active Shedded Loads:", active_load_shedded)
            print("Reactive Shedded Loads:", reactive_load_shedded)           

            active_load_shedded_by_psi = np.matmul(self.psi, np.array(active_load_shedded).reshape([-1, 1]))
            reactive_load_shedded_by_psi = np.matmul(self.psi, np.array(reactive_load_shedded).reshape([-1, 1]))

            load_shedding_penalty = float(np.dot(self.load_priority_weight, active_load_shedded_by_psi)) + \
                                        float(np.dot(self.load_priority_weight, reactive_load_shedded_by_psi))

            # 3. Renewable power curtailment penalty

            curtailed_wind_power = Pwt_cut[0]
            curtailed_solar_power = Ppv_cut[0]

            renewable_power_curtailment_penalty = self.alpha*curtailed_wind_power + self.beta*curtailed_solar_power

            # Update the prior restored load (for next control step)

            for idx, b in enumerate(self.load_bus):
                self.prior_active_restored_load[b] = active_load_restored[idx]
                self.prior_reactive_restored_load[b] = reactive_load_restored[idx]

            state = self.get_state_forecast()

            reward = (load_restoration_reward - load_shedding_penalty - renewable_power_curtailment_penalty) * REWARD_SCALING_FACTOR  

            if reward < 0:
                print()
                print("Reward is -ve")
                print("Agent Chosen Action:", action)
                print("Fuel Reserve Mapped:", fuel_reserve)
                print("SOC Reserve Mapped:", soc_reserve) 

        else:
            # Terminal state has been reached if no solution to the opf is found.

            print()
            print("OPF Infeasibility Happened")
            print("Agent Chosen MT Fuel Reserve: ", fuel_reserve)
            print("Agent Chosen ST SOC Reserve: ", soc_reserve)                
         
            state = self.terminal_state(self.num_state)
            reward = -1000000           # Very large reward

            Ppvs = state[:self.opf_num_time_steps]
            Pwtb = state[self.opf_num_time_steps: 2*self.opf_num_time_steps]  
            prior_active_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps:2*self.opf_num_time_steps+self.num_of_load]
            prior_reactive_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps+self.num_of_load: 2*self.opf_num_time_steps+2*self.num_of_load]
            Pmt = np.array([0] * self.opf_num_time_steps)
            Qmt = np.array([0] * self.opf_num_time_steps)
            Pes = np.array([0] * self.opf_num_time_steps)
            Qes = np.array([0] * self.opf_num_time_steps)
            voltages = np.zeros((self.num_of_bus, self.opf_num_time_steps))

            # TODO: Use previous states ??     
            
        # Update the timestep

        self.simulation_step += 1

        # Update the forecast/control horizon

        #self.opf_num_time_steps -= 1
    
        # Terminate the episode if the final control step is reached

        self.done = True if self.simulation_step >= CONTROL_HORIZON_LEN else False
        
        if self.debug:
            self.history['active_load_status'].append(active_load_restored)
            self.history['reactive_load_status'].append(reactive_load_restored)
            self.history['pv_power'].append(Ppvs[0])
            self.history['wt_power'].append(Pwtb[0])
            self.history['mt_power'].append([Pmt[0], Qmt[0]])
            self.history['st_power'].append([Pes[0], Qes[0]])
            self.history['voltages'].append(voltages[:,0])
            self.history['mt_remaining_fuel'].append(self.mt_remaining_fuel)
            self.history['st_soc'].append(self.es_current_storage) 
            self.history['mt_fuel_reserve'].append(fuel_reserve)
            self.history['st_soc_reserve'].append(soc_reserve)
            self.history['voltage_bus_names'] = self.bus_name        

        return state, reward, self.done, {}

    def get_opf_solution(self, control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                         prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                         wind_power_forecast, solar_power_forecast):

        # Call the Julia/JuMP OPF/MPC
        
        #opf_solution = self.opf.run_opf(control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
        #                                prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
        #                                wind_power_forecast, solar_power_forecast)

        # Call the Python/Pyomo OPF/MPC
  
        # TODO: For the purpose of performance improvement update the pyomo model parameters instead of rebuilding the model at each simulation step.

        model = self.opf.build_load_restoration_model(control_horizon, fuel_reserve, soc_reserve, prior_active_restored_load, \
                                                 prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                                 wind_power_forecast, solar_power_forecast)

        opf_solution = self.opf.compute_solution(model, self.solver, solver_tee=False, solver_options=None)

        return opf_solution
    
    def build_action_space(self):
        """
        Build the action space available to the agent to choose from.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        lower_bounds = np.array([-1.0] * 2)
        upper_bounds = np.array([1.0] * 2)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float32)

        return space

    def build_observation_space(self):
        """
        Build the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space of the environment.
        """

        dim_obs = self.num_state

        lower_bounds = np.array([0.0] * (dim_obs - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 
        upper_bounds = np.array([1.0] * dim_obs)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float)

        return space      
    
    def terminal_state(self, num_state):
        """
        Return a vector of lower of bounds of the observation space (arbitrarily chosen as terminal state).

        Parameters
        ----------
        num_state : int
            The length of the vector to return.

        Returns
        -------
        numpy.ndarray
            A vector of size num_state whose values are lower bounds of the observation space.
        """

        state = np.array([0.0] * (num_state - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 

        return state

    def get_control_history(self):

        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),            
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']), 
            'mt_fuel_reserve': np.array(self.history['mt_fuel_reserve']),
            'st_soc_reserve': np.array(self.history['st_soc_reserve']),           
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'active_load_status': np.array(self.history['active_load_status']),
            'reactive_load_status': np.array(self.history['reactive_load_status']),
            'time_stamps': self.time_of_day_list
            }

        return results


##############################################################################################################
########################################### ENV #4 ###########################################################
##############################################################################################################


class ReservePolicyEnv4(gym.Env):
    """ 
    gym environment for reserve policy learning problem.

    this env is similar to the 3rd env except that it does not build the Pyomo lr model
    at every simulation step, instead it only updates the required Pyomo model parameters.
    """  

    def __init__(self):    

        self.Zbase = 1
        self.Vbase = 4160
        self.Sbase = (self.Vbase**2)/self.Zbase
        self.w_to_kw_conv = 1000        
        
        self.wind_power_forecasts = None
        self.pv_power_forecasts = None           
        
        self.pv_max_gen = 300
        self.wind_max_gen = 400

        self.bus_name = ['650', '646', '645', '632', '633', '634', '611', '684', '671', '692', '675', '652', '680'] # IEEE 13bus balanced version
        self.num_of_bus = len(self.bus_name) 
        self.load_bus = [1, 2, 4, 5, 6, 8, 9, 10, 11]
        self.num_of_load = len(self.load_bus)  
        self.load_priority_weight = [1.0, 1.0, 0.9, 0.85, 0.8, 0.65, 0.45, 0.4, 0.3]

        self.psi = np.diag([100] * self.num_of_load)    # Penalty factor for shedding prior restored load
        self.alpha =  0.2                               # $/kWh -- penalty for wind power curtailment
        self.beta = 0.2                                 # $/kWh -- penalty for PV power curtailment

        self.es_soc_min = 20        # in %
        self.es_soc_max = 100       # in %

        self.initial_active_restored_load = [0] * self.num_of_bus 
        self.initial_reactive_restored_load = [0] * self.num_of_bus 
        
        self.es_initial_soc = 100       # in %
        self.mt_initial_fuel = 1000     # in kWh

        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        self.opf_num_time_steps = CONTROL_HORIZON_LEN    # 6h long with 5 min resolution        
        self.simulation_step = 0
        self.done = False

        self.debug = None
        self.history = None
        self.enable_debug(DEBUG)

        self.time_of_day_list = None        
        
        # Number of state variables
        self.num_state = 167         # 72 wind_forecast + 72 pv forecast + 9 active load + 9 reactive load + 5 scalar

        # Build action space        
        self.action_space = self.build_action_space()

        # Build observation space
        self.observation_space = self.build_observation_space()          
              
        # OPF model class object         
        self.opf = OPFPyomo3()        
           
        # Specify solver
        self.solver = 'glpk'   # can also be 'xpress_direct', 'cbc'     

    def enable_debug(self, debug_flag):
        self.debug = debug_flag
        if debug_flag:
            self.history = copy.deepcopy(CONTROL_HISTORY_DICT)

    def reset(self, start_index=None, init_storage=None):
        
        self.simulation_step = 0             
        self.done = False
        
        self.prior_active_restored_load = self.initial_active_restored_load
        self.prior_reactive_restored_load = self.initial_reactive_restored_load

        self.mt_remaining_fuel = self.mt_initial_fuel
        self.es_current_storage = self.es_initial_soc

        # start_index = 180
        if start_index is None:
            # 0: Index for time 07/01 00:00
            # 8856: Index for time 07/31 18:00
            start_index = np.random.randint(0, 8856)

        print()
        print("Start Index for The Renewables:", start_index)
            
        self.time_of_day_list = []        
        self.get_renewable_power_forecast_6h(start_index)

        state = self.get_state_forecast()

        return state
    
    def get_renewable_power_forecast_6h(self, start_index):        
        # Use future 6-hour forecasts as RL inputs.        
              
        os.chdir(renewable_power_forecast_dir)  
                
        forecast_reader = ForecastReader()
        available_datetimes = forecast_reader.available_dates()

        forecast_horizon = CONTROL_HORIZON_LEN 

        wind_power_forecasts = np.empty((0, forecast_horizon))
        pv_power_forecasts = np.empty((0, forecast_horizon))                        

        available_datetimes = list(available_datetimes)
        available_datetimes_for_training = available_datetimes[:8928]   # 07/01 00:00 - 07/31 23:55
        
        for datetime in available_datetimes_for_training[start_index: start_index+forecast_horizon]:
        
            renewable_power_forecast = forecast_reader.get_forecast(datetime)

            wind_power_forecast = (np.array(renewable_power_forecast['wind_gen'])*self.wind_max_gen)[:forecast_horizon]
            wind_power_forecasts = np.append(wind_power_forecasts, [wind_power_forecast], axis=0)     # in kW
            self.wind_power_forecasts = wind_power_forecasts

            pv_power_forecast = (np.array(renewable_power_forecast['pv_gen'])*self.pv_max_gen)[:forecast_horizon]
            pv_power_forecasts = np.append(pv_power_forecasts, [pv_power_forecast], axis=0)   # in kW
            self.pv_power_forecasts = pv_power_forecasts 
    
        step_time = available_datetimes_for_training[start_index]
        for _ in range(CONTROL_HORIZON_LEN):
            self.time_of_day_list.append(step_time)
            step_time += pd.Timedelta('5M')

        os.chdir(cur_dir)         

    @staticmethod
    def get_trigonomical_representation(pd_datetime):

        daily_five_min_position = STEPS_PER_HOUR * pd_datetime.hour + pd_datetime.minute / 5
        degree = daily_five_min_position / 288.0 * 2 * np.pi

        return np.sin(degree), np.cos(degree)

    def get_state_forecast(self):
        """ 
        Gather the system state at current step.

        - 13-Bus Balanced Case
        - State dimension is 167, these elements are:
            0-71: PV generation forecast for the next 6 hour
            72-143: Wind generation forecast for the next 1 hour
            144-152: Prior active restored loads (9 loads)
            153-161: Prior reactive restored loads (9 loads)
            162-166: [es_soc, mt_remaining_fuel, current_timestep, sinT, cosT] 
        """

        sim_step = self.simulation_step      
        
        pv_forecast = list(self.pv_power_forecasts[sim_step,:]*self.w_to_kw_conv/self.Sbase)
        wt_forecast = list(self.wind_power_forecasts[sim_step,:]*self.w_to_kw_conv/self.Sbase)

        current_active_load_status = []
        current_reactive_load_status = []
        for l in self.load_bus:
            current_active_load_status.append(self.prior_active_restored_load[l]*self.w_to_kw_conv/self.Sbase)
            current_reactive_load_status.append(self.prior_reactive_restored_load[l]*self.w_to_kw_conv/self.Sbase)

        for idx, p in enumerate(current_active_load_status):
            if p < 0 and abs(p) < 1e-4: # For numerical stability during RL's observation space boundary check
                current_active_load_status[idx] = 0.0 
 
        for idx, q in enumerate(current_reactive_load_status):
            if q < 0 and abs(q) < 1e-4: 
                current_reactive_load_status[idx] = 0.0  

        es_soc = self.es_current_storage/100.

        mt_fuel_remain = self.mt_remaining_fuel/self.mt_initial_fuel
        if mt_fuel_remain < 0 and abs(mt_fuel_remain) < 1e-4: 
            mt_fuel_remain = 0.0

        current_step = sim_step/CONTROL_HORIZON_LEN

        sin_t, cos_t = self.get_trigonomical_representation(self.time_of_day_list[sim_step])

        state = np.array(pv_forecast + wt_forecast + current_active_load_status + current_reactive_load_status + 
                        [es_soc, mt_fuel_remain, current_step, sin_t, cos_t])

        return state

    def step(self, action):
        """ Implementing one step of control.

        This function consists of 3 parts:
          1. Action pre-processing: convert the normalized control to their original range.        
          2. Control implementation: a. run the OPF; b. update the ES and MT status.
          3. Post-control process: gather next state, reward and done signal.

        Args:
          action: Numpy array with dimention (2,). 
            - The second element is the MT end-of-horizon fuel reserve: (-1,1) --> (0, self.mt_remaining_fuel).
            - The first element is the ES end-of-horizon SOC reserve: (-1,1) --> (20%, self.es_current_storage).
        """

        # Step 1: Pre-process actions
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map the RL agent chosen actions, at this step, to the MT fuel (in kWh) and ES SOC (in %) reserve requirements.

        fuel_reserve = self.mt_remaining_fuel*(action[0]+1)/2                                           # (-1, 1) -> (0, self.mt_remaining_fuel) MT fuel reserve
        soc_reserve = (self.es_current_storage - self.es_soc_min)*(action[1]+1)/2 + self.es_soc_min     # (-1, 1) -> (self.es_soc_min, self.es_current_storage) ES SOC reserve

        print()
        print("Simulation Step:", self.simulation_step)

        print()
        print("MT Remaining Fuel:", self.mt_remaining_fuel)
        print("ES SOC Level:", self.es_current_storage)
        print("Agent Chosen Action:", action)
        print("Fuel Reserve Mapped:", fuel_reserve)
        print("SOC Reserve Mapped:", soc_reserve) 

        # Renewable power
        p_pv= list(self.pv_power_forecasts[self.simulation_step,:]*self.w_to_kw_conv/self.Sbase)
        p_wt = list(self.wind_power_forecasts[self.simulation_step,:]*self.w_to_kw_conv/self.Sbase)      
        
        # Step 2: Control implementation - Compute OPF and Update the MT fuel and ES SOC levels
       
        opf_solution = self.get_opf_solution(fuel_reserve, soc_reserve, self.prior_active_restored_load, \
                                             self.prior_reactive_restored_load, self.mt_remaining_fuel, \
                                             self.es_current_storage, p_wt, p_pv)     

        opf_converged, P_restored, Pmt, Pes, Pwtb, Pwt_cut, Ppvs, Ppv_cut, Q_restored, Qmt, Qwtb, Qpvs, Qes, SOC_es, voltages = opf_solution
      
        if opf_converged:  
           
            # Update the MT fuel and ES SOC levels

            self.mt_remaining_fuel -= Pmt[0] * FIVE_MIN_TO_HOUR
            self.es_current_storage = SOC_es[0]

            if self.mt_remaining_fuel < 0 and abs(self.mt_remaining_fuel) < 1e-4:  # For numerical stability during Pyomo's boundary check
                self.mt_remaining_fuel = 0.0

            # Step 3: Post-control process

            # Calculate reward:
                # Consists of three parts: load restoration reward, load shedding penalty and renewable power curtailment penalty

            # 1. Load restoration reward

            active_load_restored = P_restored[:,0]      # the restored load at the current step
            reactive_load_restored = Q_restored[:,0]    # the restored load at the current step

            load_restoration_reward = np.dot(self.load_priority_weight, active_load_restored) + \
                                        np.dot(self.load_priority_weight, reactive_load_restored)
                    
            # 2. Load shedding penalty

            # Prior restored loads - re-organize by excluding non-load buses
            prior_active_restored_load_with_only_load_bus = []
            prior_reactive_restored_load_with_only_load_bus = []
            for l in self.load_bus:
                prior_active_restored_load_with_only_load_bus.append(self.prior_active_restored_load[l])
                prior_reactive_restored_load_with_only_load_bus.append(self.prior_reactive_restored_load[l])
            
            active_load_shedded = [max(0.0, (prior_active_restored_load_with_only_load_bus[idx] - active_load_restored[idx])) for idx in range(self.num_of_load)]
            reactive_load_shedded = [max(0.0, (prior_reactive_restored_load_with_only_load_bus[idx] - reactive_load_restored[idx])) for idx in range(self.num_of_load)]
            
            print()        
            print("Active Shedded Loads:", active_load_shedded)
            print("Reactive Shedded Loads:", reactive_load_shedded)           

            active_load_shedded_by_psi = np.matmul(self.psi, np.array(active_load_shedded).reshape([-1, 1]))
            reactive_load_shedded_by_psi = np.matmul(self.psi, np.array(reactive_load_shedded).reshape([-1, 1]))

            load_shedding_penalty = float(np.dot(self.load_priority_weight, active_load_shedded_by_psi)) + \
                                        float(np.dot(self.load_priority_weight, reactive_load_shedded_by_psi))

            # 3. Renewable power curtailment penalty

            curtailed_wind_power = Pwt_cut[0]
            curtailed_solar_power = Ppv_cut[0]

            renewable_power_curtailment_penalty = self.alpha*curtailed_wind_power + self.beta*curtailed_solar_power

            # Update the prior restored load (for next control step)

            for idx, b in enumerate(self.load_bus):
                self.prior_active_restored_load[b] = active_load_restored[idx]
                self.prior_reactive_restored_load[b] = reactive_load_restored[idx]

            state = self.get_state_forecast()

            reward = (load_restoration_reward - load_shedding_penalty - renewable_power_curtailment_penalty) * REWARD_SCALING_FACTOR  

            if reward < 0:
                print()
                print("Reward is -ve")
                print("Agent Chosen Action:", action)
                print("Fuel Reserve Mapped:", fuel_reserve)
                print("SOC Reserve Mapped:", soc_reserve) 

        else:
            # Terminal state has been reached if no solution to the opf is found.

            print()
            print("OPF Infeasibility Happened")
            print("Agent Chosen MT Fuel Reserve: ", fuel_reserve)
            print("Agent Chosen ST SOC Reserve: ", soc_reserve)                
         
            state = self.terminal_state(self.num_state)
            reward = -1000000           # Very large reward

            Ppvs = state[:self.opf_num_time_steps]
            Pwtb = state[self.opf_num_time_steps: 2*self.opf_num_time_steps]  
            prior_active_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps:2*self.opf_num_time_steps+self.num_of_load]
            prior_reactive_restored_load_with_only_load_bus = state[2*self.opf_num_time_steps+self.num_of_load: 2*self.opf_num_time_steps+2*self.num_of_load]
            Pmt = np.array([0] * self.opf_num_time_steps)
            Qmt = np.array([0] * self.opf_num_time_steps)
            Pes = np.array([0] * self.opf_num_time_steps)
            Qes = np.array([0] * self.opf_num_time_steps)
            voltages = np.zeros((self.num_of_bus, self.opf_num_time_steps))

            # TODO: Use previous states ??     
            
        # Update the timestep

        self.simulation_step += 1

        # Update the forecast/control horizon

        #self.opf_num_time_steps -= 1
    
        # Terminate the episode if the final control step is reached

        self.done = True if self.simulation_step >= CONTROL_HORIZON_LEN else False
        
        if self.debug:
            self.history['active_load_status'].append(active_load_restored)
            self.history['reactive_load_status'].append(reactive_load_restored)
            self.history['pv_power'].append(Ppvs[0])
            self.history['wt_power'].append(Pwtb[0])
            self.history['mt_power'].append([Pmt[0], Qmt[0]])
            self.history['st_power'].append([Pes[0], Qes[0]])
            self.history['voltages'].append(voltages[:,0])
            self.history['mt_remaining_fuel'].append(self.mt_remaining_fuel)
            self.history['st_soc'].append(self.es_current_storage) 
            self.history['mt_fuel_reserve'].append(fuel_reserve)
            self.history['st_soc_reserve'].append(soc_reserve)
            self.history['voltage_bus_names'] = self.bus_name        

        return state, reward, self.done, {}

    def get_opf_solution(self, fuel_reserve, soc_reserve, prior_active_restored_load, \
                         prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                         wind_power_forecast, solar_power_forecast):
                       
        # Update the Pyomo model parameters

        self.opf.update_load_restoration_model_parameters(fuel_reserve, soc_reserve, prior_active_restored_load, \
                                                          prior_reactive_restored_load, mt_remaining_fuel, es_prior_soc, \
                                                          wind_power_forecast, solar_power_forecast)
                       
        # Solve the model        

        opf_solution = self.opf.compute_solution(self.solver, solver_tee=False, solver_options=None)

        return opf_solution
    
    def build_action_space(self):
        """
        Build the action space available to the agent to choose from.

        Returns
        -------
        gym.spaces.Box
            The action space of the environment.
        """

        lower_bounds = np.array([-1.0] * 2)
        upper_bounds = np.array([1.0] * 2)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float32)

        return space

    def build_observation_space(self):
        """
        Build the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space of the environment.
        """

        dim_obs = self.num_state

        lower_bounds = np.array([0.0] * (dim_obs - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 
        upper_bounds = np.array([1.0] * dim_obs)

        space = spaces.Box(low=lower_bounds, 
                           high=upper_bounds, 
                           dtype=np.float)

        return space      
    
    def terminal_state(self, num_state):
        """
        Return a vector of lower of bounds of the observation space (arbitrarily chosen as terminal state).

        Parameters
        ----------
        num_state : int
            The length of the vector to return.

        Returns
        -------
        numpy.ndarray
            A vector of size num_state whose values are lower bounds of the observation space.
        """

        state = np.array([0.0] * (num_state - 5) + [0.2] + [0.0] * 2 + [-1.0] * 2) 

        return state

    def get_control_history(self):

        results = {
            'pv_power': np.array(self.history['pv_power']),
            'wt_power': np.array(self.history['wt_power']),
            'mt_power': np.array(self.history['mt_power']),
            'st_power': np.array(self.history['st_power']),            
            'mt_remaining_fuel': np.array(self.history['mt_remaining_fuel']),
            'st_soc': np.array(self.history['st_soc']), 
            'mt_fuel_reserve': np.array(self.history['mt_fuel_reserve']),
            'st_soc_reserve': np.array(self.history['st_soc_reserve']),           
            'voltages': np.array(self.history['voltages']).transpose(),
            'voltage_bus_names': np.array(self.history['voltage_bus_names']),
            'active_load_status': np.array(self.history['active_load_status']),
            'reactive_load_status': np.array(self.history['reactive_load_status']),
            'time_stamps': self.time_of_day_list
            }

        return results