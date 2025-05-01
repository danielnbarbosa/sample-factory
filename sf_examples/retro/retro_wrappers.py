import gymnasium as gym

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
    """Rewards action that moves toward the end of the floor."""
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


class LogStep(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        print(action, info, reward)
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
