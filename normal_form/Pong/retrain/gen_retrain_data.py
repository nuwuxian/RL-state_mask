import os
import PIL
import gym
import torch
import numpy as np
import copy
from PIL import Image
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN
from gym.wrappers.monitoring import video_recorder
import multiprocessing
from multiprocessing import set_start_method

G_GAE = 0.99

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



def test_mask(i_episode, env, baseline_model, mask_network, kdd_network, device):
    actions = [0,1,2,3,4,5]
    env.seed(i_episode)
    state = env.reset()
    state = grey_crop_resize(state)
    
    count = 0
    num_mask = 0

    total_reward = 0

    action_seq = []
    mask_probs = []
    values = []
    lazy_gaps = []
    kdd_probs = []

    while True:
        #vid.capture_frame()
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        baseline_dist, baseline_value = baseline_model(state)       
        #baseline_action = baseline_dist.sample().cpu().numpy()[0]
        values.append(baseline_value.data.detach().cpu().numpy()[0])
        action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])

        #Q_values = []

        #for act in actions:
        #    env_copy = copy.deepcopy(env)
        #   next_state_copy, reward_copy, _ , _ = env_copy.step(act)
        #    next_state_copy = grey_crop_resize(next_state_copy)
        #    next_state_copy = torch.FloatTensor(np.copy(next_state_copy)).unsqueeze(0).to(device)
        #    _, next_value_copy = baseline_model(next_state_copy)
        #    Q_value = reward_copy + G_GAE * next_value_copy.detach().cpu().numpy()[0]
        #    Q_values.append(Q_value)

        #lazy_gap = np.max(Q_values) - np.mean(Q_values)
        #lazy_gaps.append(lazy_gap)

        mask_dist, _ = mask_network(state)
        mask_probs.append(mask_dist.probs.detach().cpu().numpy()[0])

        kdd_dist, _ = kdd_network(state)
        kdd_probs.append(kdd_dist.probs.detach().cpu().numpy()[0])

        action_seq.append(action)

        count += 1
        print(count)
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
    
    return total_reward, count, action_seq, mask_probs, values, lazy_gaps, kdd_probs


def collect(proc_idx, env, before_model, mask_network, kdd_network, device):
    tmp_rewards = []
    tmp_counts = []

    winning_games = []
    losing_games = []


    for i in range(50):
        total_reward, count, action_seq, mask_probs, values, lazy_gaps, kdd_probs = test_mask(proc_idx* 50 + i, env, before_model, mask_network, kdd_network, device)

        if total_reward == 0:
            losing_games.append(proc_idx* 50 + i)
        else:
            winning_games.append(proc_idx* 50 + i)

        print("Test " + str(i) + " :")
        print("reward: " + str(total_reward))
        print("episode length: " + str(count))

        tmp_rewards.append(total_reward)
        tmp_counts.append(count)



        eps_len_filename = "./recording/eps_len_" + str(proc_idx* 50 + i) + ".out" 
        np.savetxt(eps_len_filename, [count])

        act_seq_filename = "./recording/act_seq_" + str(proc_idx* 50 + i) + ".out" 
        np.savetxt(act_seq_filename, action_seq)

        mask_probs_filename = "./recording/mask_probs_" + str(proc_idx* 50 + i) + ".out" 
        np.savetxt(mask_probs_filename, mask_probs)

        values_filename = "./recording/values_" + str(proc_idx* 50 + i) + ".out" 
        np.savetxt(values_filename, values)

        #lazy_gaps_filename = "./recording/lazy_gaps_" + str(proc_idx* 50 + i) + ".out" 
        #np.savetxt(lazy_gaps_filename, lazy_gaps)

        kdd_probs_filename = "./recording/kdd_probs_" + str(proc_idx* 50 + i) + ".out" 
        np.savetxt(kdd_probs_filename, kdd_probs)

    np.savetxt("./recording/reward_record_" + str(proc_idx) + ".out", tmp_rewards)
    np.savetxt("./recording/winning_game_" + str(proc_idx) + ".out", winning_games)
    np.savetxt("./recording/losing_game_" + str(proc_idx) + ".out", losing_games)

if __name__ == '__main__':

    H_SIZE = 256

    N_TESTS = 5000

    if os.path.isdir("recording"):
        os.system("rm -rf recording")


    os.system("mkdir recording")

    env = gym.make("Pong-v0").env

    num_inputs = 1
    num_outputs = env.action_space.n


    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    before_retrain = "../ppo_test/baseline/Pong-v0_+0.340_100.dat"
    PATH = "../ppo_test/checkpoints/Pong-v0_+0.850_7200.dat"
    KDD_PATH = "../kdd/ppo_test/checkpoints/Pong-v0_+0.600_450.dat"

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

    kdd_network = CNN(num_inputs, 2, H_SIZE).to(device)
    if use_cuda:
        checkpoint = torch.load(PATH)
    else:
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    kdd_network.load_state_dict(checkpoint['state_dict'])

    print("=====Generate training example=====")
    #set_start_method('spawn')

    collect_processes = [multiprocessing.Process(target=collect, kwargs={"proc_idx": i, "env": env, "before_model": before_model, "mask_network": mask_network, "kdd_network": kdd_network, "device": device}) for i in range(int(5000/50))]

    for p in collect_processes:
        p.start()
    
    for p in collect_processes:
        p.join()

    winning_games = []
    losing_games = []
    reward_records = []

    for i in range(100):
        winning_games.extend(np.loadtxt("./recording/winning_game_" + str(i) + ".out"))
        losing_games.extend(np.loadtxt("./recording/losing_game_" + str(i) + ".out"))
        reward_records.extend(np.loadtxt("./recording/reward_record_" + str(i) + ".out"))

    np.savetxt("winning_game.out", winning_games)
    np.savetxt("losing_game.out", losing_games)
    np.savetxt("reward_record.out", reward_records)