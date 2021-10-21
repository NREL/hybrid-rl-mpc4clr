"""
This script demonstrates how to call a gym env from a controller.

Action format:
0:  End_of_horizon fuel reserve; (-1, 1) -> (0, remaining_fuel_kWh) 
1:  End_of_horizon SOC reserve; (-1, 1) -> (20%, soc_level_%)

"""

import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import opf_envs  

plt.rcParams["font.family"] = "Times New Roman"

current_dir = os.getcwd()

save_dir = current_dir + "/trained_model_evaluation_results"


class ReservePolicyEnvControllerTester(object):
    """ 
    A unified controller tester for evaluating RL controller.
    """

    def __init__(self, config):
        """ 
        Initialize the controller tester.

        Args:
          config: a dictionary. Items include 'env_name' with value of gym environment name: you can use 
            'ReservePolicy-v0'. 'start_idx' is the starting index of the episode to be simulated. 
            E.g., 0 is the index for time 07/01 00:00.
          Example:
            config = {'env_name': 'ReservePolicy-v0', 'start_idx': 0}
        """

        env_name, start_idx, init_storage, env_config = self.parse_config(config)
        if env_name is None:
            raise ValueError("You must set the env_name in the configuration")

        self.env = gym.make(env_name)
        self.env.enable_debug(True)
        self.env_state = self.env.reset(start_idx, init_storage)
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
        Update the configuration and reset the environment.
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
        #gen_df = pd.DataFrame(gen_data, columns=['time', 'mt_p', 'mt_q', 'st_p', 'st_q', 'pv_p', 'wt_p', 
        #                                         'mt_remaining_fuel', 'st_soc', 'mt_fuel_reserve', 'st_soc_reserve'])

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

    def plot_control_result(self,
                            process_heat_map=True,
                            active_power_profile=True,
                            reactive_power_profile=True,
                            bus_voltage=True,
                            remaining_fuel=True,
                            reserve_level=True,
                            data_source=None):
        """ Plot the load restoration process under the control implemented.

        Args:
          process_heat_map: A Boolean, plot restoration process heat map if true.
          active_power_profile: A Boolean, plot active power profiles of all generators if true.
          reactive_power_profile: A  Boolean, plot reactive power profiles of all generators if true.
          bus_voltage: A Boolean, plot bus voltage profiles of all buses if true.
          remaining_fuel: A Boolean, plot storage and micro-turbine remaining fueal if true.
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

        if bus_voltage:
            lower_vol_limit = 0.95
            upper_vol_limit = 1.05
            lb_volts = np.array([lower_vol_limit] * 72)
            ub_volts = np.array([upper_vol_limit] * 72)

            plt.figure(figsize=(5, 3), dpi=200)
            for k in range(voltage_history.shape[0]):

                if '.1' in voltage_bus_names[k]:
                    color = 'k'
                elif '.2' in voltage_bus_names[k]:
                    color = 'r'
                else:
                    color = 'b'

                plt.plot(voltage_history[k], linewidth=1, color=color, alpha=0.5, label=voltage_bus_names[k])
                #if min(voltage_history[k]) < lower_vol_limit:
                #    print(voltage_history[k][64:])
                #    print(voltage_bus_names[k])
                #    print(np.mean(voltage_history[k][64:]))
                #    print(min(voltage_history[k]))

            plt.plot(lb_volts, linewidth=1, linestyle='dashed', color='red')
            plt.plot(ub_volts, linewidth=1, linestyle='dashed', color='red')

            self._plot_util('Time', 'Voltages (pu)', start_timestamp, (0.0, 1.06), if_grid=True)

        if active_power_profile:
            plt.figure(figsize=(5, 3), dpi=200)

            total_generation_p = [np.array([pv_p[idx], wt_p[idx], st_p[idx], mt_p[idx]]).sum(axis=0)
                                  for idx in range(72)]

            plt.plot(pv_p, color='#DB4437', linewidth=1, alpha=0.95, label='PV')
            plt.plot(wt_p, color='#4285F4', linewidth=1, alpha=0.95, label='WT')
            plt.plot(st_p, color='#0F9D58', linewidth=1, alpha=0.95, label='ST')
            plt.plot(mt_p, color='#F4B400', linewidth=1, alpha=0.95, label='MT')

            plt.plot(total_generation_p, color='k', linewidth=1, alpha=0.8, label='Total Gen')

            self._plot_util('Time', 'Active Power (kW)', start_timestamp, (-200, 800),
                            'upper left', if_grid=True)

        if reactive_power_profile:
            plt.figure(figsize=(5, 3), dpi=200)

            total_generation_q = [np.array([st_q[idx], mt_q[idx]]).sum(axis=0)
                                  for idx in range(72)]

            plt.plot(st_q, color='#0F9D58', linewidth=1, alpha=0.95, label='ST')
            plt.plot(mt_q, color='#F4B400', linewidth=1, alpha=0.95, label='MT')

            plt.plot(total_generation_q, color='k', linewidth=1, alpha=0.8, label='Total Gen')

            self._plot_util('Time', 'Reactive Power (kvar)', start_timestamp, (-200, 800), 'upper left', if_grid=True)

        if remaining_fuel:
            plt.figure(figsize=(5, 3), dpi=200)

            plt.plot(mt_fuel, label='MT fuel')
            plt.plot(st_soc, label='ST SOC')
            plt.plot([0.128] * len(st_soc), label='ST SOC Minimum', linestyle='dashed', color='red')

            self._plot_util('Time', 'Energy Remained (Normalized)', start_timestamp, (-0.5, 1.05),
                            'upper center', if_grid=True)

        if reserve_level:
            plt.figure(figsize=(5, 3), dpi=200)

            plt.plot(mt_fuel_reserve, label='MT fuel reserve')
            plt.plot(st_soc_reserve, label='ST SOC reserve')
            
            self._plot_util('Time', 'Energy Reserved (Normalized)', start_timestamp, (-0.5, 1.05),
                            'upper center', if_grid=True)

        if process_heat_map:
            plt.figure(figsize=(5, 3), dpi=200)

            plt.pcolor(load_history.transpose())

            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Load 1-9, decreasing importance', fontsize=12)
            plt.tick_params(labelsize=12)
            plt.xticks(range(0, 73, 12), [str((start_timestamp.hour + int(a / 12)) % 24) +
                                          ':' + (str(start_timestamp.minute) if start_timestamp.minute > 5
                                                 else '0' + str(start_timestamp.minute)) for a in range(0, 73, 12)])
            plt.yticks([x + 0.5 for x in range(self.env.num_of_load) if x % 2 == 0],
                       [str(a + 1) for a in range(self.env.num_of_load) if a % 2 == 0])
            plt.tight_layout()

        plt.show()

    @staticmethod
    def _plot_util(x_content, y_content, start_timestamp, ylim, legend_pos=None, ncol=2, if_grid=False):

        plt.xlabel(x_content, fontsize=12)
        plt.ylabel(y_content, fontsize=12)
        plt.tick_params(labelsize=12)
        plt.xticks(range(0, 73, 12), [str((start_timestamp.hour + int(a / 12)) % 24) +
                                      ':' + (str(start_timestamp.minute) if start_timestamp.minute > 5
                                             else '0' + str(start_timestamp.minute)) for a in range(0, 73, 12)])
        plt.ylim(ylim)

        if legend_pos is not None:
            plt.legend(loc=legend_pos, ncol=ncol, fontsize=8)

        plt.tight_layout()
        if if_grid:
            plt.grid()