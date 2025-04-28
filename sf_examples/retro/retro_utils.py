import sys
import os

# Go up one level and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../stable-retro')))

from typing import Optional

import gymnasium as gym
import retro

from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
    RewardScalingWrapper,
)

from gymnasium.wrappers.transform_reward import TransformReward
from sf_examples.retro.retro_wrappers import ForwardMovementRewardWrapper, LogStep
from sf_examples.retro.retro_discretizer import AirStrikeDiscretizer, KungFuDiscretizer

RETRO_W = RETRO_H = 84
LEFT_ACTION_INDEX = 1
RIGHT_ACTION_INDEX = 2


class RetroSpec:
    def __init__(self, name, env_id, discretizer, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.discretizer = discretizer
        self.has_timer = False


RETRO_ENVS = [
    RetroSpec("Airstriker-Genesis", "Airstriker-Genesis", AirStrikeDiscretizer),
    RetroSpec("KungFu-Nes", "KungFu-Nes", KungFuDiscretizer),
]


def retro_env_by_name(name):
    for cfg in RETRO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Retro env")


def make_retro_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    retro_spec = retro_env_by_name(env_name)

    if cfg.state is None:
        env = retro.make(retro_spec.env_id, render_mode=render_mode)
    else:
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode)

    env = retro_spec.discretizer(env)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(env.action_space)

    if retro_spec.default_timeout is not None:
        env._max_episode_steps = retro_spec.default_timeout

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    #env = EpisodicLifeEnv(env)
    # noinspection PyUnresolvedReferences
    #if "FIRE" in env.unwrapped.get_action_meanings():
    #    env = FireResetEnv(env)
    #env = ClipRewardEnv(env)
    #env = RewardScalingWrapper(env, 0.01)
    env = TransformReward(env, lambda r: r - 0.02)  # small penalty per timestep
    env = ForwardMovementRewardWrapper(env, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, movement_reward=0.1)  # big reward for moving in the correct direction depending on floor
    #env = LogStep(env)
    env = gym.wrappers.ResizeObservation(env, (RETRO_W, RETRO_H))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)
    #print(env.observation_space.sample())

    return env
