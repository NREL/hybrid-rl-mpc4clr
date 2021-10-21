import os
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env

from opf_envs.envs.gym_reserve_policy import ReservePolicyEnv
from opf_envs.envs.gym_reserve_policy import ReservePolicyEnv2
from opf_envs.envs.gym_reserve_policy import ReservePolicyEnv3
from opf_envs.envs.gym_reserve_policy import ReservePolicyEnv4

current_file_path = os.getcwd()
LOG_PATH = os.path.join(current_file_path, 'results/')  # Local
#LOG_PATH = os.path.join('/scratch/aeseye/', 'rl_mpc_reserve_policy/')  # HPC

if __name__ == "__main__":

    ## Command line args

    parser = argparse.ArgumentParser(description="Training RLlib agents to learn power system reserves")
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--name-env", type=str, default="ReservePolicy-v0")
    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--redis-password", type=str, default=None)
    args = parser.parse_args()

    ## Register the ReservePolicy RL environment
  
    env_name = "ReservePolicyEnv-v0"
    register_env(env_name, lambda config: ReservePolicyEnv())

    ## Initialize Ray

    ray.init(_temp_dir="/tmp/scratch/ray")      # HPC
    #ray.init(_node_ip_address='192.168.0.10')  # local, add your IP if NREL VPN is ON (during local running)
    num_cpus = args.num_cpus - 1

    ## Run TUNE Experiments (Train the RL Agent)
    
    tune.run(
        args.algo,
        name=env_name,     
        stop={"time_total_s": 3600},  # or: stop={"training_iteration": 100}
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
