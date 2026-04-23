import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer

class D4PGAgent:
    def __init__(self, obs_dim=11, action_dim=3, device='cuda',
                 gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-3,
                 n_atoms=51, v_min=-10, v_max=10, batch_size=256):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer()
        self.atoms = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def select_action(self, obs, noise=0.1):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -0.15, 0.15)

    def update(self):
        if len(self.buffer) < 2000:
            return None, None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Distributional Bellman update
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            next_logits = self.critic_target(next_obs, next_actions)
            next_probs = torch.softmax(next_logits, dim=1)

            # Project onto shifted support
            Tz = rewards + (1 - dones) * self.gamma * self.atoms.unsqueeze(0)
            Tz = Tz.clamp(self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long().clamp(0, self.n_atoms - 1)
            u = b.ceil().long().clamp(0, self.n_atoms - 1)

            target_probs = torch.zeros_like(next_probs)
            target_probs.scatter_add_(1, l, next_probs * (u.float() - b))
            target_probs.scatter_add_(1, u, next_probs * (b - l.float()))

        # Critic loss
        logits = self.critic(obs, actions)
        critic_loss = -(target_probs * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor loss
        actor_loss = -torch.softmax(self.critic(obs, self.actor(obs)), dim=1).mul(self.atoms).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update targets
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])