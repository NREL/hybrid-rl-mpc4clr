"""
Call a gym env from a controller.

Action format:
0:  End_of_horizon fuel reserve; (-1, 1) -> (0, remaining_fuel_kWh) 
1:  End_of_horizon SOC reserve; (-1, 1) -> (20%, soc_level_%)

"""

import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import opf_mpc_envs  

current_dir = os.getcwd()

save_dir = current_dir + "/trained_model_evaluation_results"


class ReservePolicyEnvControllerTester(object):
    """ 
    Tester for evaluating the trained RL-MPC controller.
    """

    def __init__(self, config):
        """ 
        Initialize the controller tester.

        Args:
          config: a dictionary. Items include 'env_name' with value of gym environment name: you can use 
          'RLMPCReservePolicyEnv-v0'. 'start_idx' is the starting index of the episode to be simulated: 
          e.g., 0 is the index for time 07/01 00:00.
          Example:
            config = {'env_name': 'RLMPCReservePolicyEnv-v0', 'start_idx': 0}
        """

        env_name, start_idx, init_storage, env_config = self.parse_config(config)
        if env_name is None:
            raise ValueError("You must set the env_name in the configuration")

        self.env = gym.make(env_name)
        self.env.enable_debug(True)
        self.env_state = self.env.reset(start_index=start_idx, init_storage=init_storage)       
        self.episode_finished = False
        self.episode_reward = 0.0
        self.episode_reward_history = []

    @staticmethod
    def parse_config(config):
        """ 
        Parse the configuration.
        """

        KEYS = ('env_name', 'start_idx', 'init_storage', 'env_config')
        results = []

        for key in KEYS:
            try:
                results.append(config[key])
            except KeyError:
                results.append(None)

        return results
    
    def update_config_and_reset(self, config):
        """ 
        Update the configuration and reset the RL-MPC environment.
        """

        env_name, start_idx, init_storage, env_config = self.parse_config(config)
        self.env_state = self.env.reset(start_idx, init_storage)
        self.episode_finished = False
        self.episode_reward = 0.0
        self.episode_reward_history = []

    def apply_control(self, action_vec):
        """ 
        Implement one step of control.

        Args:
          action_vec: a list. See top of this script for requirement.
        """

        single_step_results = self.env.step(action_vec)

        self.env_state = single_step_results[0]
        self.episode_reward += single_step_results[1]
        self.episode_reward_history.append(single_step_results[1])
        self.episode_finished = single_step_results[2]
       
    def save_control_result(self, path):
        
        if not os.path.isdir(path):
            os.makedirs(path)

        results = self.env.get_control_history()

        gen_label = ['mt_power', 'st_power', 'pv_power', 'wt_power', 'mt_remaining_fuel', 'st_soc', 'mt_fuel_reserve', 'st_soc_reserve']
        gen_data = np.hstack([results[label].reshape((72, -1)) for label in gen_label])
        time_label = np.array(results['time_stamps']).reshape((72, -1))
        gen_data = np.hstack((time_label, gen_data))
        
        pv_power = results['pv_power']       
        pv_power_df = pd.DataFrame(pv_power)
        pv_power_df.to_csv(path + '/pv_power_results.csv', index=False)

        wt_power = results['wt_power']
        wt_power_df = pd.DataFrame(wt_power)
        wt_power_df.to_csv(path + '/wt_power_results.csv', index=False)
        
        mt_power_p, mt_power_q = results['mt_power'][:, 0], results['mt_power'][:, 1]
        mt_power_p_df, mt_power_q_df = pd.DataFrame(mt_power_p), pd.DataFrame(mt_power_q)
        mt_power_p_df.to_csv(path + '/mt_power_p_results.csv', index=False)
        mt_power_q_df.to_csv(path + '/mt_power_q_results.csv', index=False)

        st_power_p, st_power_q = results['st_power'][:, 0], results['st_power'][:, 1]
        st_power_p_df, st_power_q_df = pd.DataFrame(st_power_p), pd.DataFrame(st_power_q)
        st_power_p_df.to_csv(path + '/st_power_p_results.csv', index=False)
        st_power_q_df.to_csv(path + '/st_power_q_results.csv', index=False)

        mt_remaining_fuel = results['mt_remaining_fuel']
        mt_remaining_fuel_df = pd.DataFrame(mt_remaining_fuel)
        mt_remaining_fuel_df.to_csv(path + '/mt_remaining_fuel_results.csv', index=False)

        st_soc = results['st_soc']
        st_soc_df = pd.DataFrame(st_soc)
        st_soc_df.to_csv(path + '/st_soc_results.csv', index=False)

        mt_fuel_reserve = results['mt_fuel_reserve']
        mt_fuel_reserve_df = pd.DataFrame(mt_fuel_reserve)
        mt_fuel_reserve_df.to_csv(path + '/mt_fuel_reserve_results.csv', index=False)

        st_soc_reserve = results['st_soc_reserve']
        st_soc_reserve_df = pd.DataFrame(st_soc_reserve)
        st_soc_reserve_df.to_csv(path + '/st_soc_reserve_results.csv', index=False)

        volt = results['voltages'].transpose()
        volt_df = pd.DataFrame(volt, columns=results['voltage_bus_names'])
        volt_df.to_csv(path + '/voltage_results.csv', index=False)

        active_load = results['active_load_status']
        active_load_df = pd.DataFrame(active_load, columns=['load_' + str(x + 1) for x in range(active_load.shape[1] )])
        active_load_df.to_csv(path + '/active_load_results.csv', index=False)        

        reactive_load = results['reactive_load_status']
        reactive_load_df = pd.DataFrame(reactive_load, columns=['load_' + str(x + 1) for x in range(reactive_load.shape[1] )]) 
        reactive_load_df.to_csv(path + '/reactive_load_results.csv', index=False)

    def plot_control_result(self, data_source=None):
        """ 
        Plot the load restoration process under the control implemented.
        """

        if data_source is None:

            results = self.env.get_control_history()

            start_timestamp = results['time_stamps'][0]

            pv_p = results['pv_power']
            wt_p = results['wt_power']
            mt_p, mt_q = results['mt_power'][:, 0], results['mt_power'][:, 1]
            st_p, st_q = results['st_power'][:, 0], results['st_power'][:, 1]
            mt_fuel = results['mt_remaining_fuel']
            st_soc = results['st_soc']
            mt_fuel_reserve = results['mt_fuel_reserve']
            st_soc_reserve = results['st_soc_reserve']

            voltage_history = results['voltages']
            voltage_bus_names = results['voltage_bus_names']

            load_history = results['active_load_status']

            print("Episodic reward (Total): %f." % self.episode_reward)

        else:
            gen_df = pd.read_csv(os.path.join(data_source, 'gen_results.csv'))
            start_timestamp = pd.to_datetime(gen_df.time[0])
            pv_p, wt_p, mt_p, st_p = gen_df.pv_p, gen_df.wt_p, gen_df.mt_p, gen_df.st_p
            mt_q, st_q = gen_df.mt_q, gen_df.st_q
            mt_fuel = gen_df.mt_remaining_fuel
            st_soc = gen_df.st_soc
            mt_fuel_reserve = gen_df.mt_fuel_reserve
            st_soc_reserve = gen_df.st_soc_reserve

            volt_df = pd.read_csv(os.path.join(data_source, 'volt_results.csv'))
            voltage_history = volt_df.to_numpy().transpose()
            voltage_bus_names = volt_df.columns.to_list()

            load_df = pd.read_csv(os.path.join(data_source, 'load_results.csv'))
            load_history = load_df.to_numpy()

        # Bus voltage plots:

        lower_vol_limit = 0.95
        upper_vol_limit = 1.05
        lb_volts = np.array([lower_vol_limit] * 72)
        ub_volts = np.array([upper_vol_limit] * 72)

        plt.figure(figsize=(12, 6), dpi=200)
        for k in range(voltage_history.shape[0]):

            if '.1' in voltage_bus_names[k]:
                color = 'k'
            elif '.2' in voltage_bus_names[k]:
                color = 'r'
            else:
                color = 'b'

            plt.plot(voltage_history[k], linewidth=1, color=color, alpha=0.5, label=voltage_bus_names[k])               
            plt.plot(lb_volts, linewidth=1, linestyle='dashed', color='red')
            plt.plot(ub_volts, linewidth=1, linestyle='dashed', color='red')
            plt.xlabel('Time')
            plt.ylabel('Voltages (pu)')
            plt.legend()
            plt.grid()
        plt.show()

        # Active power plots:

        total_generation_p = [np.array([pv_p[idx], wt_p[idx], st_p[idx], mt_p[idx]]).sum(axis=0) 
                                for idx in range(72)]

        plt.figure(figsize=(12, 6), dpi=200)
        plt.plot(pv_p, color='#DB4437', linewidth=1, alpha=0.95, label='PV')
        plt.plot(wt_p, color='#4285F4', linewidth=1, alpha=0.95, label='WT')
        plt.plot(st_p, color='#0F9D58', linewidth=1, alpha=0.95, label='ST')
        plt.plot(mt_p, color='#F4B400', linewidth=1, alpha=0.95, label='MT')
        plt.plot(total_generation_p, color='k', linewidth=1, alpha=0.8, label='Total Gen')
        plt.xlabel('Time')
        plt.ylabel('Active Power (kW)')
        plt.legend()
        plt.grid()
        plt.show()        

        # Reactive power plots:

        total_generation_q = [np.array([st_q[idx], mt_q[idx]]).sum(axis=0) 
                                for idx in range(72)]

        plt.figure(figsize=(12, 6), dpi=200)
        plt.plot(st_q, color='#0F9D58', linewidth=1, alpha=0.95, label='ST')
        plt.plot(mt_q, color='#F4B400', linewidth=1, alpha=0.95, label='MT')
        plt.plot(total_generation_q, color='k', linewidth=1, alpha=0.8, label='Total Gen')
        plt.xlabel('Time')
        plt.ylabel('Reactive Power (kvar)')
        plt.legend()
        plt.grid()
        plt.show()
        
        # Remaining fuel plots:

        Cbat = 800  # battery capacity in kWh

        plt.figure(figsize=(12, 6), dpi=200)
        plt.plot(mt_fuel, label='MT fuel')
        plt.plot(Cbat*np.array(st_soc)/100, label='ST SOC')        
        plt.xlabel('Time')
        plt.ylabel('Energy Remained (kWh)')
        plt.legend()
        plt.grid()
        plt.show()

        # Reserve level plots:

        plt.figure(figsize=(12, 6), dpi=200)
        plt.plot(mt_fuel_reserve, label='MT fuel reserve')
        plt.plot(Cbat*np.array(st_soc_reserve)/100, label='ST SOC reserve')
        plt.xlabel('Time')
        plt.ylabel('Energy Reserved (kWh)')
        plt.legend()
        plt.grid()
        plt.show()  

        # Restored load plots:  
         
        restored_loads = load_history.transpose() 
        load_labels = ['Load1', 'Load2', 'Load3', 'Load4', 'Load5', 'Load6', 'Load7', 'Load8', 'Load9']      
                
        plt.figure(figsize=(12, 6), dpi=200)
        for l in range(len(restored_loads)):
            plt.plot(restored_loads[l], label=load_labels[l])
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Restored Loads (kW) - Priority of Loads decreases from Load1 to 9', fontsize=12)
        plt.tick_params(labelsize=12)
        plt.legend()
        plt.grid()       
        plt.show()