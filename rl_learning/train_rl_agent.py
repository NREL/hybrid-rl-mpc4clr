"""
Reinforcement Learning (RL) Agent Training 
"""

import os
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from opf_mpc_envs.envs.gym_mpc_env import ReservePolicyEnv

current_file_path = os.getcwd()
LOG_PATH = os.path.join(current_file_path, 'results/')  

if __name__ == "__main__":

    ## Command line args

    parser = argparse.ArgumentParser(description="Training RLlib agents to learn power system reserves")
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--name-env", type=str, default="RLMPCReservePolicy-v0")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--redis-password", type=str, default=None)
    args = parser.parse_args()

    ## Register the RL-MPC environment
  
    env_name = "RLMPCReservePolicyEnv-v0"
    register_env(env_name, lambda config: ReservePolicyEnv())

    ## Initialize Ray

    ray.init()  
    num_cpus = args.num_cpus - 1

    ## Run TUNE Experiments (Train the RL Agent)
    
    tune.run(
        args.algo,
        name=env_name,     
        stop={"time_total_s": 3600},  # 1h (3600s) training. Or: stop={"training_iteration": 100}
        checkpoint_freq=10,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=3,
        local_dir=LOG_PATH,
        config={
            "env": env_name,
            "num_workers": args.num_cpus, 
            "num_gpus": args.num_gpus,
            "ignore_worker_failures": True
            }
        )