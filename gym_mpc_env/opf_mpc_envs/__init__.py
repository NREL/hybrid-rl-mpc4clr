"""
RL environments for designing power system reserve learning tasks.
"""

from gym.envs.registration import register

register(
        id='RLMPCReservePolicy-v0',
        entry_point='opf_mpc_envs.envs:ReservePolicyEnv' 
        )