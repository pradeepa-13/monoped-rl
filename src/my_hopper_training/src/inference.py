#!/usr/bin/env python3

import rospy
import gym
import monoped_env
from stable_baselines3 import SAC

rospy.init_node('monoped_inference', anonymous=True, log_level=rospy.INFO)

env = gym.make('Monoped-v0')

model = SAC.load("/root/monoped_ws/models/monoped_phase1_stand_b")

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

# import rospy
# import gym
# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# import monoped_env
# from d4pg.agent import D4PGAgent

# rospy.init_node('monoped_inference', anonymous=True)
# env = gym.make('Monoped-v0')

# agent = D4PGAgent(obs_dim=11, action_dim=3, device='cpu')
# agent.load('/root/monoped_ws/models/d4pg/best_model.pt')

# obs = env.reset()
# while True:
#     action = agent.select_action(obs, noise=0.0)  # deterministic
#     obs, reward, done, _ = env.step(action)
#     if done:
#         obs = env.reset()