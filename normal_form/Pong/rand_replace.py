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
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN
from gym.wrappers.monitoring import video_recorder


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

env = gym.make("Pong-v0").env

H_SIZE = 256
num_inputs = 1
num_outputs = env.action_space.n

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BASELINE_PATH = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
PATH = "./ppo_test/checkpoints/Pong-v0_+0.910_20170.dat"

baseline_model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(BASELINE_PATH)
else:
    checkpoint = torch.load(BASELINE_PATH, map_location=torch.device('cpu'))
baseline_model.load_state_dict(checkpoint['state_dict'])

#PATH = "./ppo_test/checkpoints/Pong-v0_+19.600_7380.dat"
mask_network = CNN(num_inputs, 2, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(PATH)
else:
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
mask_network.load_state_dict(checkpoint['state_dict'])



critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")

avg_rewards = []

for j in range(6):
    replay_rewards = []
    for i_episode in range(500):
        action_sequence_path = "./recording/act_seq_" + str(i_episode) + ".out"
        recorded_actions = np.loadtxt(action_sequence_path)

        iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
        iteration_ends = np.loadtxt(iteration_ends_path)

        critical_frames_start = critical_steps_starts[i_episode]
        critical_frames_end = critical_steps_ends[i_episode]

        random_replacement_steps = critical_frames_end - critical_frames_start

        start_range = int(iteration_ends - random_replacement_steps)
        
        rand_frames_start = np.random.choice(start_range)
        rand_frames_end = rand_frames_start + random_replacement_steps
    
        env.seed(i_episode)
        state = env.reset()
        state = grey_crop_resize(state)
        
        count = 0
    
        done = False
        total_reward = 0
    
    
        while not done:
    
            state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
    
            if count < rand_frames_start:
                next_state, reward, done, _ = env.step(int(recorded_actions[count]))
            
            elif count <= rand_frames_end:
                next_state, reward, done, _ = env.step(np.random.choice(6))
            
            else:
                baseline_dist, _ = baseline_model(state)
                mask_dist, _ = mask_network(state)
                #baseline_action = baseline_dist.sample().cpu().numpy()[0]
                baseline_action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])
    
                #mask_action = mask_dist.sample().cpu().numpy()[0]
                mask_action = np.argmax(mask_dist.probs.detach().cpu().numpy()[0])
    
            
                if mask_action == 1:
                    action = baseline_action
                else:
                    action = np.random.choice(6)
                    
                next_state, reward, done, _ = env.step(action)
    
            count += 1
            
            done = reward
    
            next_state = grey_crop_resize(next_state)
            state = next_state
            total_reward += reward
        
        if total_reward == 1:
            replay_rewards.append(1)
        else:
            replay_rewards.append(0)
    

    avg_rewards.append(np.mean(replay_rewards))
    print("Replay Round " + str(j) + " :")
    print("Current average winning rate: ", np.mean(replay_rewards))

print("Final mean: ", np.mean(avg_rewards))
print("STD: ", np.std(avg_rewards))
