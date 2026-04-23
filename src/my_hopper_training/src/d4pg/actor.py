import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim=11, action_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * 0.15  # scale to action space ±0.15