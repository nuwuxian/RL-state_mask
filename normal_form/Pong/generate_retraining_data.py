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



def test_mask(i_episode, env, baseline_model, mask_network, device):
    #vid_path = "./recording/vid_mask_" + str (i_episode) + ".mp4"
    #vid = video_recorder.VideoRecorder(env,path=vid_path)

    env.seed(i_episode)
    state = env.reset()
    state = grey_crop_resize(state)
    
    count = 0
    num_mask = 0

    total_reward = 0

    action_seq = []
    mask_probs = []

    while True:
        #vid.capture_frame()
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        baseline_dist, _ = baseline_model(state)       
        #baseline_action = baseline_dist.sample().cpu().numpy()[0]
        action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])


        mask_dist, _ = mask_network(state)
        mask_probs.append(mask_dist.probs.detach().cpu().numpy()[0])

        action_seq.append(action)

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
    
    return total_reward, count, num_mask, action_seq, mask_probs

H_SIZE = 256

N_TESTS = 10000

if os.path.isdir("recording"):
    os.system("rm -rf recording")


os.system("mkdir recording")

env = gym.make("Pong-v0").env

num_inputs = 1
num_outputs = env.action_space.n


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

before_retrain = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"

PATH = "./ppo_test/masknet/Pong-v0_+0.898_19660.dat"

before_model = CNN(num_inputs, num_outputs, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(before_retrain)
else:
    checkpoint = torch.load(before_retrain, map_location=torch.device('cpu'))
before_model.load_state_dict(checkpoint['state_dict'])


mask_network = CNN(num_inputs, 2, H_SIZE).to(device)
if use_cuda:
    checkpoint = torch.load(PATH)
else:
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
mask_network.load_state_dict(checkpoint['state_dict'])



print("=====Generate training example=====")

tmp_rewards = []
tmp_counts = []
tmp_num_masks = []
winning_games = []
losing_games = []


for i in range(N_TESTS):
    total_reward, count, num_mask, action_seq, mask_probs = test_mask(i, env, before_model, mask_network, device)

    if total_reward == 0:
        losing_games.append(i)
    else:
        winning_games.append(i)

    print("Test " + str(i) + " :")
    print("reward: " + str(total_reward))
    print("episode length: " + str(count))
    print("num of masks: " + str(num_mask))
    tmp_rewards.append(total_reward)
    tmp_counts.append(count)
    tmp_num_masks.append(num_mask)



    eps_len_filename = "./recording/eps_len_" + str(i) + ".out" 
    np.savetxt(eps_len_filename, [count])

    act_seq_filename = "./recording/act_seq_" + str(i) + ".out" 
    np.savetxt(act_seq_filename, action_seq)

    mask_probs_filename = "./recording/mask_probs_" + str(i) + ".out" 
    np.savetxt(mask_probs_filename, mask_probs)

np.savetxt("reward_record.out", tmp_rewards)
np.savetxt("winning_game.out", winning_games)
np.savetxt("losing_game.out", losing_games)

