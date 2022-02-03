"""
Test the trained RL control policy for one episode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import pickle

import gym
import opf_mpc_envs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from opf_mpc_envs.envs.gym_mpc_env import ReservePolicyEnv
from rl_controller_evaluation import ReservePolicyEnvControllerTester

current_file_path = os.getcwd()

REWARD_SCALING_FACTOR = 0.001

def get_rl_controller(run_class, env, checkpoint):
    """ 
    Import controller from checkpoint.
    """
    
    ray.init()    

    # Load configuration from file

    config_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory.")
    
    print("config_path:", config_path)
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    if "num-cpus" in config:
        config["num-cpus"] = min(1, config["num-cpus"])
   
    cls = get_agent_class(run_class)
    agent = cls(env=env, config=config)
    agent.restore(checkpoint)

    return agent

if __name__ == "__main__":
    
    current_file_path = os.getcwd()
    
    run_class = 'PPO'
    env_id = 'RLMPCReservePolicy-v0'
    checkpoint = os.path.join(current_file_path,
         'results/RLMPCReservePolicyEnv-v0/PPO_RLMPCReservePolicyEnv-v0_64772_00000_0_2022-02-01_13-20-34/checkpoint_000002/checkpoint-2')
     
    env = gym.make(env_id)  
    
    register_env(env_id, lambda config: env)
    
    rl_agent = get_rl_controller(run_class, env_id, checkpoint)
    
    config = {'env_name': env_id,
              'start_idx': None}
    controller_tester = None

    # Testing multiple scenarios with randomly generated IDs/indices.
    
    num_test_scenarios = 10
    scenario_ids = [np.random.randint(0, 8856) for _ in range(num_test_scenarios)]
    reward_list = []
    
    for s_id in scenario_ids:
        
        print("start_idx", s_id)

        config['start_idx'] = s_id
        
        if controller_tester is None:
            controller_tester = ReservePolicyEnvControllerTester(config)
        else:
            controller_tester.update_config_and_reset(config)

        while not controller_tester.episode_finished:
            action = rl_agent.compute_action(controller_tester.env_state)
            controller_tester.apply_control(action)

        reward_list.append(controller_tester.episode_reward)

print("Reward list", reward_list)
print("Average reward level is: %f" % np.mean(reward_list))

# Reward list over scenarios plot:

fig = plt.figure(figsize=(10,5))
plt.plot(np.array(reward_list), linewidth = 3, marker='o')
plt.grid()
plt.xlabel("Scenarios", fontsize=15)
plt.ylabel("Episode Reward Mean", fontsize=15)
plt.tick_params(axis="x", labelsize=15)
plt.tick_params(axis="y", labelsize=15)
plt.xticks(range(0, num_test_scenarios, 1), [str(int(c)) for c in range(0, num_test_scenarios, 1)])
plt.show()

# Learning curve plot:

progress = os.path.join(current_file_path,
                'results/RLMPCReservePolicyEnv-v0/PPO_RLMPCReservePolicyEnv-v0_64772_00000_0_2022-02-01_13-20-34/progress.csv')

progress_df = pd.read_csv(progress)

episode_reward_mean = progress_df['episode_reward_mean']

fig = plt.figure(figsize=(10,5))
plt.plot(episode_reward_mean, linewidth = 3)
plt.yticks(fontsize=15)
plt.xlabel("Training Iterations", fontsize=15)
plt.ylabel("Episode Reward Mean", fontsize=15)
plt.title("Reserve Policy Learning Curve", fontsize=15)
plt.grid()
plt.show()

# Plots control results for the last scenario tested

print("Plot the last scenario tested.")

controller_tester.plot_control_result()