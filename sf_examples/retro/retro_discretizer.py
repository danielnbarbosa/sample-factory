"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
"""

import gymnasium as gym
import numpy as np

import retro


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                ["LEFT"],
                ["RIGHT"],
                ["LEFT", "DOWN"],
                ["RIGHT", "DOWN"],
                ["DOWN"],
                ["DOWN", "B"],
                ["B"],
            ],
        )


class AirStrikeDiscretizer(Discretizer):

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],
                ["LEFT"],
                ["RIGHT"],
                ["LEFT", "B"],
                ["RIGHT", "B"],
                ["B"],
            ],
        )


class KungFuDiscretizer(Discretizer):

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],
                ["LEFT"],
                ["RIGHT"],
                ["UP"],
                ["DOWN"],
                ["B"],
                ["UP", "B"],
                ["UP", "LEFT", "B"],
                ["UP", "RIGHT", "B"],
                ["DOWN", "LEFT", "B"],
                ["DOWN", "RIGHT", "B"],
                ["DOWN", "A"],
            ],
        )


class DoubleDragonDiscretizer(Discretizer):

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],
                ["LEFT"],
                ["RIGHT"],
                ["UP"],
                ["DOWN"],
                ["A"],  # punch
                ["B"],  # kick
                ["A", "B"],  # jump
                ["A", "B", "LEFT"],
                ["A", "B", "RIGHT"],
            ],
        )


class SuperMarioBrosDiscretizer(Discretizer):

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],
                ["LEFT"],
                ["RIGHT"],
#                ["UP"],
#                ["DOWN"],
                ["A"],  # jump
#                ["B"],  # run, fireball
                ["RIGHT", "A"],
                ["RIGHT", "B"],
                ["RIGHT", "A", "B"],
                ["LEFT", "A"],
            ],
        )


def main():
    env = retro.make(
        game="SonicTheHedgehog-Genesis",
        use_restricted_actions=retro.Actions.DISCRETE,
    )
    print("retro.Actions.DISCRETE action_space", env.action_space)
    env.close()

    env = retro.make(game="SonicTheHedgehog-Genesis")
    env = SonicDiscretizer(env)
    print("SonicDiscretizer action_space", env.action_space)
    env.close()


if __name__ == "__main__":
    main()
