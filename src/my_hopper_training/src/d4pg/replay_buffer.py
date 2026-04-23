import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (np.array(obs), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_obs), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)