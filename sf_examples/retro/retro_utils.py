import sys
import os

# Go up one level and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../stable-retro')))

from typing import Optional

import gymnasium as gym
import retro

from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
    RewardScalingWrapper,
)

from gymnasium.wrappers.transform_reward import TransformReward
from sf_examples.retro.retro_wrappers import ClimbReward, LogKungFu, LogDoubleDragon, LogSuperMarioBros, CropObservation, EpisodicLifeEnv , EvalKungFu, EvalDoubleDragon, EvalSuperMarioBros
from sf_examples.retro.retro_discretizer import AirStrikeDiscretizer, KungFuDiscretizer, DoubleDragonDiscretizer, SuperMarioBrosDiscretizer

RETRO_H = 84
RETRO_W = 84

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
    RetroSpec("DoubleDragon-Nes", "DoubleDragon-Nes", DoubleDragonDiscretizer),
    RetroSpec("SuperMarioBros-Nes", "SuperMarioBros-Nes", SuperMarioBrosDiscretizer),
]


def retro_env_by_name(name):
    for cfg in RETRO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Retro env")


def make_retro_env_old(env_name, cfg, env_config, render_mode: Optional[str] = None):
    retro_spec = retro_env_by_name(env_name)

    #if cfg.state is None:
    #    env = retro.make(retro_spec.env_id, render_mode=render_mode)
    #else:
    if cfg.mode == 'train':
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario")
    else:
        print("Using scenario-eval.json")
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario-eval")

    env = retro_spec.discretizer(env)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(env.action_space)

    if retro_spec.default_timeout is not None:
        env._max_episode_steps = retro_spec.default_timeout

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    if cfg.mode =='train':
        print("EpisodicLifeEnv() enabled.")
        env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    #env = RewardScalingWrapper(env, 0.01)
    #env = TransformReward(env, lambda r: r - 0.02)  # small penalty per timestep
    #env = TransformReward(env, lambda r: r + 0.4)  # reward per timestep
    if env_name == 'DoubleDragon-Nes':
        env = ClimbReward(env, target_action=3, target_reward=0.075)
        env = ClimbReward(env, target_action=4, target_reward=-0.075)

    if cfg.mode == 'eval' and env_name == 'KungFu-Nes':
        env = EvalKungFu(env)
    if cfg.mode == 'eval' and env_name == 'DoubleDragon-Nes':
        env = EvalDoubleDragon(env)
    if cfg.mode == 'eval' and env_name == 'SuperMarioBros-Nes':
        env = EvalSuperMarioBros(env)

    if cfg.mode == 'log' and env_name == 'DoubleDragon-Nes':
        env = LogDoubleDragon(env)
    if cfg.mode == 'log' and env_name == 'SuperMarioBros-Nes':
        env = LogSuperMarioBros(env)
    #env = CropObservation(env, top=64, left=0, height=112, width=240)  # KungFu crop
    #env = CropObservation(env, top=0, left=0, height=190, width=240)  # DoubleDragon crop
    env = gym.wrappers.ResizeObservation(env, (RETRO_H, RETRO_W))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)
    #print(env.observation_space.sample())

    return env


def make_retro_env_kung_fu(env_name, cfg, env_config, render_mode: Optional[str] = None):
    retro_spec = retro_env_by_name(env_name)

    if cfg.mode == 'train':
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario")
    else:
        print("Using scenario-eval.json")
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario-eval")

    env = retro_spec.discretizer(env)

    if retro_spec.default_timeout is not None:
        env._max_episode_steps = retro_spec.default_timeout

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    #env = CropObservation(env, top=64, left=0, height=112, width=240)
    env = gym.wrappers.ResizeObservation(env, (RETRO_H, RETRO_W))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)

    if cfg.mode =='train': env = EpisodicLifeEnv(env); print("EpisodicLifeEnv() enabled.")
    if cfg.mode == 'eval': env = EvalKungFu(env)
    if cfg.mode == 'log': env = LogKungFu(env)

    return env



def make_retro_env_double_dragon(env_name, cfg, env_config, render_mode: Optional[str] = None):
    retro_spec = retro_env_by_name(env_name)

    if cfg.mode == 'train':
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario")
    else:
        print("Using scenario-eval.json")
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario-eval")

    env = retro_spec.discretizer(env)

    if retro_spec.default_timeout is not None:
        env._max_episode_steps = retro_spec.default_timeout

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = ClimbReward(env, target_action=3, target_reward=0.075)
    env = ClimbReward(env, target_action=4, target_reward=-0.075)
    #env = CropObservation(env, top=0, left=0, height=190, width=240)
    env = gym.wrappers.ResizeObservation(env, (RETRO_H, RETRO_W))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)

    if cfg.mode =='train': env = EpisodicLifeEnv(env); print("EpisodicLifeEnv() enabled.")
    if cfg.mode == 'eval': env = EvalDoubleDragon(env)
    if cfg.mode == 'log': env = LogDoubleDragon(env)

    return env



def make_retro_env_super_mario_bros(env_name, cfg, env_config, render_mode: Optional[str] = None):
    retro_spec = retro_env_by_name(env_name)

    if cfg.mode == 'train':
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario")
    else:
        print("Using scenario-eval.json")
        env = retro.make(retro_spec.env_id, state=cfg.state, render_mode=render_mode, scenario="scenario-eval")

    env = retro_spec.discretizer(env)

    if retro_spec.default_timeout is not None:
        env._max_episode_steps = retro_spec.default_timeout

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (RETRO_H, RETRO_W))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)

    if cfg.mode =='train': env = EpisodicLifeEnv(env); print("EpisodicLifeEnv() enabled.")
    if cfg.mode == 'eval': env = EvalSuperMarioBros(env)
    if cfg.mode == 'log': env = LogSuperMarioBros(env)

    return env


# choose which function to use
make_retro_env = make_retro_env_super_mario_bros
