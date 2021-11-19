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

register(
        id='ReservePolicy-v4',
        entry_point='opf_envs.envs:ReservePolicyEnv5'
        )

register(
        id='ReservePolicy-v5',
        entry_point='opf_envs.envs:ReservePolicyEnv6'
        )

register(
        id='ReservePolicy-v6',
        entry_point='opf_envs.envs:ReservePolicyEnv7'
        )

register(
        id='ReservePolicy-v7',
        entry_point='opf_envs.envs:ReservePolicyEnv8'
        )

register(
        id='ReservePolicy-v9',
        entry_point='opf_envs.envs:ReservePolicyEnv10'
        )

register(
        id='ReservePolicy-v10',
        entry_point='opf_envs.envs:ReservePolicyEnv11'
        )

register(
        id='ReservePolicy-v11',
        entry_point='opf_envs.envs:ReservePolicyEnv12'
        )

register(
        id='ReservePolicy-v12',
        entry_point='opf_envs.envs:ReservePolicyEnv13'
        )
