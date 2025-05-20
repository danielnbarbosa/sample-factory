import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Union
import time

class ActionRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_action: int, target_reward: float = 0.01):
        super().__init__(env)
        self.target_action = target_action
        self.target_reward = target_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply target_reward if the chosen action matches
        if action == self.target_action:
            reward += self.target_reward

        #print(action, info, reward)
        return obs, reward, terminated, truncated, info


class ForwardActiontReward(gym.Wrapper):
    """
    Rewards action that moves toward the end of the floor.
    This was used earlier in KungFu but was not as effective as tracking x_pos directly.
    """
    def __init__(self, env, left_action: int, right_action: int, movement_reward: float = 0.01):
        super().__init__(env)
        self.left_action = left_action
        self.right_action = right_action
        self.movement_reward = movement_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        floor = info['floor']  # floor is 0 indexed, so [0,4]
        if floor % 2 == 0 and action == self.left_action:  # if even floor
            reward += self.movement_reward
        if floor % 2 == 1 and action == self.right_action:  # if odd floor
            reward += self.movement_reward

        #print(action, info, reward)
        return obs, reward, terminated, truncated, info


class ClimbReward(gym.Wrapper):
    """
    Rewards up action while climbing.
    Used in DoubleDragon to climb fences and ladders.
    """
    def __init__(self, env, target_action: int, target_reward: float = 0.1):
        super().__init__(env)
        self.target_action = target_action
        self.target_reward = target_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Apply target_reward if the chosen action matches
        y_status = info['y_status']
        if action == self.target_action and y_status == 2:
            reward += self.target_reward

        #print(action, info, reward)
        return obs, reward, terminated, truncated, info


class LogKungFu(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if abs(reward) > 0:
            print(action, info, reward)

        return obs, reward, terminated, truncated, info


class LogDoubleDragon(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        keys_to_watch = ['x_pos', 'x_pos_player', 'y_pos', 'lives', 'health', 'enemy1_health', 'enemy2_health', 'mission', 'part', 'section', 'screen', 'time']
        info_sub = {key: info[key] for key in keys_to_watch if key in info}
        if abs(reward) > 0:
            print(action, info_sub, reward)

        return obs, reward, terminated, truncated, info


class LogSuperMarioBros(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.combos={
                0: "NOOP          ",
                1: "LEFT          ",
                2: "RIGHT         ",
#                "UP",
#                "DOWN",
                3: "JUMP          ",
#                "RUN",
                4: "RIGHT JUMP    ",
                5: "RIGHT RUN     ",
                6: "RIGHT JUMP RUN",
                7: "LEFT JUMP     ",
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if abs(reward) > 0:
            print(self.combos[action], info, reward)
        #time.sleep(1)

        return obs, reward, terminated, truncated, info




class EvalKungFu(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.evals = []
        self.n_evals_to_run = 5

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1
        if info['lives'] == 0:
            dragon = info['dragon']
            floor = info['floor']
            score = (dragon * 5) + (floor + 1)
            print(f"{dragon}-{floor}", score)
            self.evals.append(score)
            if len(self.evals) == self.n_evals_to_run:
                print(self.evals)
                print(min(self.evals), sum(self.evals) / self.n_evals_to_run, max(self.evals))
                self.env.close()

        return obs, reward, terminated, truncated, info


class EvalDoubleDragon(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.map_stage_to_score = {
        '1-1-1' : 0,
        '1-1-2' : 1,
        '1-1-3' : 2,
        '1-1-4' : 3,
        '1-1-5' : 4,
        '1-2-1' : 5,
        '1-2-2' : 6,
        '2-1-1' : 7,
        '2-1-2' : 8,
        '2-1-3' : 9,
        '2-1-4' : 10,
        '3-1-1' : 11,
        '3-1-2' : 12,
        '3-1-3' : 13,
        '3-1-4' : 14,
        '3-1-5' : 15,
        '3-1-6' : 16,
        '3-2-1' : 17,
        '3-2-2' : 18,
        '3-2-3' : 19,
        '3-2-4' : 20,
        '3-2-5' : 18,  # ropers in incorrect lower path
        '3-2-6' : 18,  # after ropers
        '3-2-7' : 18,  # top of incorrect lower path
        '3-3-1' : 21,
        '3-3-2' : 22,
        '3-3-3' : 23,
        '3-3-4' : 24,
        '3-3-5' : 25,
        '3-3-6' : 26,
        '3-4-1' : 27,
        '3-4-2' : 28,
        '4-1-1' : 29,
        }


        self.evals = []
        self.n_evals_to_run = 5

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1
        if info['lives'] == -1:
            stage = f"{info['mission'] + 1}-{info['part'] + 1}-{info['section'] + 1}"
            score = self.map_stage_to_score[stage]
            print(f"Stage: {stage}    Score: {score}     Steps:{self.steps}")
            self.evals.append(score)
            if len(self.evals) == self.n_evals_to_run:
                print(self.evals)
                print(f"{min(self.evals)}    {sum(self.evals) / self.n_evals_to_run}    {max(self.evals)}      {self.steps / self.n_evals_to_run}")
                self.env.close()

        return obs, reward, terminated, truncated, info


class EvalSuperMarioBros(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.evals = []
        self.n_evals_to_run = 10

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps += 1
        if info['lives'] == -1:
            levelHi = info['levelHi']
            levelLo = info['levelLo']
            score = (levelHi * 4) + (levelLo)
            print(f"World: {levelHi + 1}-{levelLo + 1}    Score: {score}    Steps:{self.steps}")
            self.evals.append(score)
            if len(self.evals) == self.n_evals_to_run:
                print("")
                print(self.evals)
                print("min    avg    max    steps")
                print(f" {min(self.evals)}     {sum(self.evals) / self.n_evals_to_run}     {max(self.evals)}     {self.steps / self.n_evals_to_run}")
                self.env.close()

        return obs, reward, terminated, truncated, info


class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, top, left, height, width):
        super().__init__(env)
        self.top = top
        self.left = left
        self.height = height
        self.width = width

        orig_shape = env.observation_space.shape  # (H, W, C)
        new_shape = (height, width, orig_shape[2])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs[self.top:self.top+self.height, self.left:self.left+self.width, :]


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    :param env: the environment to wrap

    Modified from sample-factory version to work with stable-retro games.
    Assumes info dicts has 'lives' key.
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated | truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info["lives"]
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)
        self.lives = info["lives"]
        return obs, info