import gym
import numpy as np
from stable_baselines3 import DQN

class IDSEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.idx = 0
        self.action_space = gym.spaces.Discrete(2)  # normal or attack
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
    def reset(self):
        self.idx = 0
        return self.X[self.idx]
    def step(self, action):
        reward = 1 if action == self.y[self.idx] else -1
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros(self.X.shape[1])
        return obs, reward, done, {}

def train_rl(X_train, y_train, timesteps=5000):
    env = IDSEnv(X_train, y_train)
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model
