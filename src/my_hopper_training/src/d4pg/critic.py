import torch
import torch.nn as nn

class Critic(nn.Module):
    """Distributional critic — outputs N_ATOMS logits over value distribution."""
    def __init__(self, obs_dim=11, action_dim=3, hidden=256, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = torch.linspace(v_min, v_max, n_atoms)

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_atoms)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)  # logits over atom distribution

    def get_atoms(self, device):
        return self.atoms.to(device)