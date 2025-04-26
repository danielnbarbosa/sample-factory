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
    

class ForwardMovementRewardWrapper(gym.Wrapper):
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