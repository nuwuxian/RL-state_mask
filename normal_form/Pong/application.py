import os, sys, tqdm
sys.path.append('..')
import torch, gym
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from explainer.gp_utils import VisualizeCovar
from pong.utils import rl_fed, NNPolicy, prepro


# crop resize
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


def run_exploration(method, budget, num_trajs, num_step=3, fix_importance=True, random_importance=False):
    tie = []
    win = []
    correct_trajs_all = []
    num_loss = 0
    loss_seeds = []

    win_lose_path = "./recording/reward_record.out"
    win_lose = np.loadtxt(win_lose_path).astype('int32')

    for i in range(num_trajs):
        if method == 'mask_net':
            mask_probs_path = "./recording/mask_probs_" + str(i) + ".out"
            mask_probs = np.loadtxt(mask_probs_path)
            confs = mask_probs[:,1]
        elif method == 'value_max':
            value_path = "./recording/value_seq_" + str(i) + ".out"
            confs = np.loadtxt(value_path)
        
        iteration_ends_path = "./recording/eps_len_" + str(i) + ".out"
        iteration_ends = int(np.loadtxt(iteration_ends_path))

        orin_reward = 1000 if win_lose[i] == 1 else -1000

        if orin_reward == 1000:
            continue
        seed = i
        loss_seeds,append(seed)
        importance_traj = np.argpartition(confs, -num_step)[-num_step:]  # Indices not sorted

        j = 0
        j_1 = 0
        correct_trajs = []
        for _ in range(budget):
            replay_reward_10, traj = run_exploration_traj(env_name=env_name, seed=seed, model=model,
                                                          original_traj=original_traj, max_ep_len=max_ep_len,
                                                          importance=importance_traj, render=False)
            if replay_reward_10 == 0:
                j += 1
            if replay_reward_10 == 1:
                j_1 += 1
            if replay_reward_10 == 1 and len(correct_trajs) == 0:
                correct_trajs.append(traj)
        correct_trajs_all.append(correct_trajs)
        tie.append(j)
        win.append(j_1)
        num_loss += 1
    print(num_loss)

    obs_all = []
    acts_all = []
    for trajs in correct_trajs_all:
        for traj in trajs:
            for step in range(len(traj[0])):
                obs_all.append(traj[0][step].numpy())
                acts_all.append(traj[1][step])

    obs_all = np.array(obs_all)
    acts_all = np.array(acts_all)

    print(obs_all.shape)
    print(acts_all.shape)

    return np.array(tie), np.array(win), correct_trajs_all, obs_all, acts_all, loss_seeds


def run_exploration_traj(env_name, seed, model, traj_len, importance, max_ep_len=200, render=False):

    acts_orin = original_traj['actions']
    start_step = 0

    env = gym.make("Pong-v0").env


    episode_length, epr, done = 0, 0, False  # bookkeeping
   

    state = env.reset()
    state = grey_crop_resize(state)
    act_set = np.array([0, 1, 2, 3, 4, 5])
    state_all = []
    action_all = []
    for i in range(traj_len + 100):
        if epr != 0:
            break
        # Steps before the important steps reproduce original traj.
        if start_step+i in importance:
            state_all.append(state)
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])

        # Important steps take random actions.
        if start_step + i in importance:
            act_set_1 = act_set[act_set!=action]
            action = np.random.choice(act_set_1)
            action_all.append(action)
        # Steps after the important steps take optimal actions.
        obs, reward, done, expert_policy = env.step(action)
        state = grey_crop_resize(obs)
        if render: env.render()
        epr += reward
        # save info!
        episode_length += 1
    
    return epr, (state_all, action_all)


def run_patch_traj(env_name, seed, model, obs_dict, act_dict, p, max_ep_len=200, eps=1e-4,
                   render=False, mix_policy=True):

    
    env = gym.make("Pong-v0").env
    in_dict = False

    act_idx = np.random.binomial(1, p)
    episode_length, epr, done = 0, 0, False  # bookkeeping
    state = env.reset()
    state = grey_crop_resize(state)

    for i in range(max_ep_len):
        if epr != 0:
            break
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])
        # check the lookup table and take the corresponding action if state is similar.
        state_diff = np.sum(np.abs(obs_dict - state.numpy()), (1, 2, 3))
        if np.min(state_diff) < eps:
            in_dict = True
            if mix_policy:
                idx = np.argmin(state_diff)
                actions = [action, act_dict[idx]]
                # act_idx = np.random.binomial(1, p)
                action = actions[act_idx]
        state, reward, done, expert_policy = env.step(action)
        state = grey_crop_resize(obs)
        if render: env.render()
        epr += reward
        # save info!
        episode_length += 1
    # print(episode_length)
    return epr, in_dict

save_path = 'exp_results/'
env = gym.make("Pong-v0").env

num_inputs = 1
num_outputs = env.action_space.n

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
baseline = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
model = CNN(num_inputs, num_outputs, H_SIZE).to(device)

if use_cuda:
    checkpoint = torch.load(baseline)
else:
    checkpoint = torch.load(baseline, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# Patch individual trajs and policy.
def patch_trajs_policy(exp_method, budget, num_patch_traj, num_test_traj, free_test=False, collect_dict=True):
    print(exp_method)
    if collect_dict:
        if exp_method == 'dgp':
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(exp_method, budget, sal, num_patch_traj)
        elif exp_method == 'saliency':
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(exp_method, budget, sal, num_patch_traj,
                                                                                   fix_importance=False,
                                                                                   random_importance=True)
        else:
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(exp_method, budget, sal, num_patch_traj,
                                                                                   fix_importance=False,
                                                                                   random_importance=False)
    else:
        tie = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['tie']
        win = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['win']
        obs_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['obs']
        acts_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['acts']
        loss_seeds = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['seed']
   
    total_trajs_num = float(win.shape[0])
    win_num = np.count_nonzero(win)
    print('Win rate: %.2f' % (100 * (win_num / total_trajs_num)))
    print('Exploration success rate: %.2f' % (100 * (np.mean(win) / budget)))
   
    # print(obs_dict.shape)
    # print(acts_dict.shape)
    # print(len(loss_seeds)) 
    num_seed_trajs = int((len(loss_seeds)/num_patch_traj)*obs_dict.shape[0])
    loss_seeds = loss_seeds[0:num_seed_trajs]
    obs_dict = obs_dict[0:num_seed_trajs, ]
    acts_dict = acts_dict[0:num_seed_trajs, ]

    # print(len(loss_seeds))
    # print(obs_dict.shape)
    # print(acts_dict.shape)

    # Get the patch prob.
    num_rounds = 0
    num_loss = 0
    for i in range(num_test_traj):
        seed = i + 1000
        r_1, in_dict = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=0, max_ep_len=200, eps=1e-3,
                                      render=False, mix_policy=False)
        if r_1 !=0 and in_dict:
            num_rounds += 1.0
            if r_1 == -1:
                num_loss += 1.0
    p = num_loss/num_rounds
    print('===')
    print(p)
    print('===')
    num_rounds = 0
    results_1 = []
    results_p = []
    for i in range(num_test_traj):
        if i % 100 == 0:
            print(i)
        if i < len(loss_seeds) and not free_test:
            seed = int(loss_seeds[i])
        else:
            seed = i
        # print('=========') 
        r_1, _ = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=0, max_ep_len=200, eps=1e-3,
                                render=False, mix_policy=False)
        # print(r_1)
        # print('----')
        r_p, _ = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=p, max_ep_len=200, eps=1e-5,
                                render=False, mix_policy=True)
        # print(r_p)
        if r_1 != 0 and r_p !=0:
            num_rounds += 1
            results_1.append(r_1)
            results_p.append(r_p)

    results_1 = np.array(results_1)
    results_p = np.array(results_p)

    num_win_1 = np.where(results_1==1)[0].shape[0]
    num_win_p = np.where(results_p==1)[0].shape[0]

    win_diff = results_1 - results_p
    num_all_win = np.where(win_diff==0)[0].shape[0]
    num_1_win_p_loss = np.where(win_diff==2)[0].shape[0]
    num_1_loss_p_win = np.where(win_diff==-2)[0].shape[0]

    print('Testing winning rate of the original model %.2f' % (100 * (num_win_1/num_rounds)))
    print('Testing winning rate of the patched model %.2f' % (100 * (num_win_p/num_rounds)))
    print('Total Number of games: %d' % num_rounds)
    print('Number of games that original policy wins but patched policy loses: %d' % num_1_win_p_loss)
    print('Number of games that original policy loses but patched policy win: %d' % num_1_loss_p_win)
   
    np.savez(save_path+exp_method+'_patch_results_'+str(budget)+'.npz', tie=tie, win=win,
             obs=obs_dict, acts=acts_dict, results_1=results_1, results_p=results_p, seed=loss_seeds, p=p)

    return 0


budget = 10
num_patch_traj = 1880
num_test_traj = 500

exp_methods = ['mask_net', 'value_max']

for k in range(2):
    patch_trajs_policy(exp_methods[k], sals[k], budget, num_patch_traj, num_test_traj, free_test=True, collect_dict=True)