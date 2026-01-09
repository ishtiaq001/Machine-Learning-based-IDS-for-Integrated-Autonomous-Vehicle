import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.preprocessing import maybe_transpose

class IDSEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.idx = 0
        self.n_features = X.shape[1]
        self.action_space = gym.spaces.Discrete(len(np.unique(y)))  # adapt to number of classes
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

    def reset(self):
        self.idx = 0
        return self.X[self.idx]

    def step(self, action):
        # Reward shaping: +2 for correct, -1 for incorrect
        reward = 2 if action == self.y[self.idx] else -1
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros(self.n_features, dtype=np.float32)
        return obs, reward, done, {}

def train_rl(X_train, y_train, timesteps=20000):
    """
    Improvements:
    - More timesteps for better training (20k)
    - Reward shaping (+2/-1)
    - Multi-class support
    - Normalize and wrap env for SB3
    """
    # Wrap in vectorized env for SB3
    env = DummyVecEnv([lambda: IDSEnv(X_train, y_train)])

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=20000,       # larger buffer
        learning_starts=500,     # more initial exploration
        batch_size=64,           # larger batch
        train_freq=8,            # less frequent updates
        target_update_interval=500,
        gamma=0.99,              # discount factor
        exploration_fraction=0.3, # longer exploration
        exploration_final_eps=0.02,
        verbose=1
    )

    model.learn(total_timesteps=timesteps)
    return model
