"""
RL-MPC simulation/learning environment tester.
"""

import os
import gym
import opf_mpc_envs

current_dir = os.getcwd()
os.chdir(os.path.dirname(current_dir) + "/rl_learning")

env = gym.make("RLMPCReservePolicy-v0")


class TestRLMPCEnv:

    def test_reset(self):

        num_state_vars = 40
        
        obs0 = env.reset()

        assert obs0.shape[0] == num_state_vars

        for state_val in obs0:
            assert state_val >= 0.0 and state_val <= 1.0

    def test_observation_space(self):

        num_state_vars = 40

        observation_space = env.observation_space

        assert observation_space.shape[0] == num_state_vars

        state = observation_space.sample()          # random states

        for state_val in state:
            assert state_val >= 0.0 and state_val <= 1.0
            
    def test_action_space(self):

        num_actions = 2
        
        action_space = env.action_space

        assert action_space.shape[0] == num_actions

        action = action_space.sample()               # random actions

        for action_val in action:
            assert action_val >=-1.0 and action_val<=1.0    

    def test_step(self):

        num_state_vars = 40

        env.reset()

        action = env.action_space.sample()  

        state, reward, done, _ = env.step(action)

        assert len(state) == num_state_vars
        
        for state_val in state:
            assert state_val >= 0.0 and state_val <= 1.0
          

# pytest test_rl_mpc_env.py