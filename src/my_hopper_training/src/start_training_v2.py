#!/usr/bin/env python3

'''
    Original Training code made by Ricardo Tellez <rtellez@theconstructsim.com>
    Moded by Miguel Angel Rodriguez <duckfrost@theconstructsim.com>
    Visit our website at www.theconstructsim.com
'''
import gym
import os
import time
import numpy
import random
from gym import wrappers
from std_msgs.msg import Float64
# ROS packages required
import rospy
import rospkg


# import our training environment
import monoped_env
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C
from stable_baselines3 import SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer

if __name__ == '__main__':
    
    rospy.init_node('monoped_gym', anonymous=True, log_level=rospy.INFO)

    # Create the Gym environment
    env = gym.make('Monoped-v0')
    # check_env(env)

    # Create the RL Agent
    # n_actions = 3
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

    os.makedirs("/root/monoped_ws/models", exist_ok=True)    

    # model = SAC("MlpPolicy", env, verbose=1,
    #         tensorboard_log="/root/monoped_ws/logs/",
    #         ent_coef=0.01,          # encourage exploration
    #         learning_rate=3e-4,
    #         batch_size=256)
    
    checkpoint_cb = CheckpointCallback(
    save_freq=25000,
    save_path="/root/monoped_ws/models/",
    name_prefix="monoped_FINAL_HOP",
    verbose=2
    )

    model = SAC.load("/root/monoped_ws/models/monoped_FINAL_HOP_50000_steps",
         env=env,
         custom_objects={
             "ent_coef": 0.15,
             "learning_rate": 2e-4,
             "action_space": env.action_space
         })

    model.replay_buffer = ReplayBuffer(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=1,
        optimize_memory_usage=model.replay_buffer.optimize_memory_usage,
    )

    model.learn(total_timesteps=200000, log_interval=4, callback=checkpoint_cb)
    model.save("/root/monoped_ws/models/monoped_FINAL_HOP")
    model.save_replay_buffer("/root/monoped_ws/models/monoped_FINAL_HOP_buffer")

    # model = SAC("MlpPolicy", env, verbose=1,
    #         tensorboard_log="/root/monoped_ws/logs/",
    #         ent_coef=0.3,
    #         learning_rate=3e-4,
    #         batch_size=256)

    # del model # remove to demonstrate saving and loading
    # model = TD3.load("rl_monoped")
    # model = A2C.load("rl_monoped")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    
    env.close()
