import gym
import numpy as np
from ppo_lasso import Masknet
import os
import torch as T
from stable_baselines3 import PPO

if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    eta_origin = -0.5738
    agent = PPO.load("./baseline/Pendulum-v0")

    print(agent.get_parameters())


