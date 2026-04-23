#!/usr/bin/env python3
import rospy
import gym
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import monoped_env
from d4pg.agent import D4PGAgent
from torch.utils.tensorboard import SummaryWriter

rospy.init_node('monoped_d4pg', anonymous=True, log_level=rospy.INFO)

env = gym.make('Monoped-v0')
agent = D4PGAgent(obs_dim=11, action_dim=3, device='cpu', v_min=-100, v_max=200)
writer = SummaryWriter('/root/monoped_ws/logs/d4pg')

os.makedirs('/root/monoped_ws/models/d4pg', exist_ok=True)

total_steps = 0
best_reward = -np.inf

for episode in range(3000):
    obs = env.reset()
    episode_reward = 0
    done = False

    while not done:
        noise = max(0.08, 0.5 - episode * 0.0002)  # starts high, decays to 0.05
        action = agent.select_action(obs, noise=noise)
        next_obs, reward, done, _ = env.step(action)
        agent.buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += reward
        total_steps += 1

        critic_loss, actor_loss = agent.update()

    writer.add_scalar('reward/episode', episode_reward, episode)
    if critic_loss is not None:
        writer.add_scalar('loss/critic', critic_loss, episode)
        writer.add_scalar('loss/actor', actor_loss, episode)

    print(f"Episode {episode} | Reward: {episode_reward:.1f} | Steps: {total_steps} | Noise: {noise:.3f}")

    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save('/root/monoped_ws/models/d4pg/best_model.pt')

    if episode % 500 == 0:
        agent.save(f'/root/monoped_ws/models/d4pg/checkpoint_{episode}.pt')

env.close()
writer.close()