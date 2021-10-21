"""
RL environments for designing power system reserve learning tasks.
"""

from gym.envs.registration import register

register(
        id='ReservePolicy-v0',
        entry_point='opf_envs.envs:ReservePolicyEnv'
        )

register(
        id='ReservePolicy-v1',
        entry_point='opf_envs.envs:ReservePolicyEnv2'
        )

register(
        id='ReservePolicy-v2',
        entry_point='opf_envs.envs:ReservePolicyEnv3'
        )

register(
        id='ReservePolicy-v3',
        entry_point='opf_envs.envs:ReservePolicyEnv4'
        )