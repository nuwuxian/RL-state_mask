import os
import PIL
import gym
import torch
import base64
import imageio
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
#from pyvirtualdisplay import Display
#from IPython.display import clear_output
from torch.distributions import Categorical
#from IPython import display as ipythondisplay
# from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN
from gym.wrappers.monitoring import video_recorder

threshold = 0.99

def make_env():    # this function creates a single environment
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _thunk():
        env = gym.make("Pong-v0").env
        return env
    return _thunk

def grey_crop_resize(state): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left = 0
    top = 34  # empirically chosen
    right = 160
    bottom = 194  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d # C*H*W

def test_baseline(i_episode, env, baseline_model, device):
    env.seed(i_episode)
    state = env.reset()
    state = grey_crop_resize(state)
    
    count = 0


    total_reward = 0
    while True:
        #vid.capture_frame()
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        baseline_dist, _ = baseline_model(state)
        #baseline_action = baseline_dist.sample().cpu().numpy()[0]
        baseline_action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])

        action = baseline_action

        count += 1
        next_state, reward, done, _ = env.step(action)

        if reward == -1:
            break
        elif reward == 1:
            total_reward += reward
            break

        done = reward

        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward
    
    return total_reward, count

def attack(i_episode, env, baseline_model, mask_network, device):

    env.seed(i_episode)
    state = env.reset()
    state = grey_crop_resize(state)
    
    count = 0
    total_reward = 0
    attack_num = 0

    while True:
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        baseline_dist, _ = baseline_model(state)
        mask_dist, _ = mask_network(state)
        #baseline_action = baseline_dist.sample().cpu().numpy()[0]
        baseline_action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])

        #mask_action = mask_dist.sample().cpu().numpy()[0]
        mask_action = np.argmax(mask_dist.probs.detach().cpu().numpy()[0])
        mask_prob = mask_dist.probs.detach().cpu().numpy()[0]
        
        if mask_prob[1] > threshold:
            action = np.random.choice(6)
            attack_num += 1
        else:
            action = baseline_action

        count += 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        if total_reward != 0:
            break
        next_state = grey_crop_resize(next_state)
        state = next_state
        
    
    if total_reward == 1:
        total_reward = 1
    else:
        total_reward = 0
    

    return total_reward, count, attack_num

H_SIZE = 256
N_TESTS = 500


env = gym.make("Pong-v0").env

num_inputs = 1
num_outputs = env.action_space.n


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

baseline = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"


baseline_model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(baseline)
else:
    checkpoint = torch.load(baseline, map_location=torch.device('cpu'))
baseline_model.load_state_dict(checkpoint['state_dict'])

PATH = "./ppo_test/masknet/Pong-v0_+0.910_20170.dat"
mask_network = CNN(num_inputs, 2, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(PATH)
else:
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
mask_network.load_state_dict(checkpoint['state_dict'])

tmp_rewards = []
tmp_counts = []

print("=====Test original model=====")
for i in range(N_TESTS):
    total_reward, count = test_baseline(i+10000, env, baseline_model, device)
    print("Test " + str(i) + " :")
    print("reward: " + str(total_reward))
    print("episode length: " + str(count))
    tmp_rewards.append(total_reward)
    print("current reward mean ", np.mean(tmp_rewards))
    tmp_counts.append(count)


tmp_rewards2 = []
tmp_counts2 = []
attack_nums = []

print("=====Test model after attack=====")
for i in range(N_TESTS):
    total_reward, count, attack_num = attack(i+10000, env, baseline_model, mask_network, device)
    print("Test " + str(i) + " :")
    print("reward: " + str(total_reward))
    print("episode length: " + str(count))
    tmp_rewards2.append(total_reward)
    print("current reward mean ", np.mean(tmp_rewards2))
    tmp_counts2.append(count)
    print("attack steps: ", attack_num)
    attack_nums.append(attack_num)

print("Average winning rate before: ", np.mean(tmp_rewards))
print("Average winning rate after: ", np.mean(tmp_rewards2))
print("Average attack steps: ", np.mean(attack_nums))

