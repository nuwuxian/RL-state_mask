import os, sys
from tqdm import tqdm, trange
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np


def select_critical_steps(importance, percentage = 0.3):

    k = int(200 * percentage)
    idx = importance[:k]
    idx.sort()

    critical_steps_start = idx[0]
    critical_steps_end = idx[0]

    ans = 0
    count = 0

    tmp_end = idx[0]
    tmp_start = idx[0]

    for i in range(1, len(idx)):
     
        # Check if the current element is
        # equal to previous element +1
        if idx[i] == idx[i - 1] + 1:
            count += 1
            tmp_end = idx[i]
             
        # Reset the count
        else:
            count = 1
            tmp_start = idx[i]
            tmp_end = idx[i]
             
        # Update the maximum
        if count > ans:
            ans = count
            critical_steps_start = tmp_start
            critical_steps_end = tmp_end
    
    return np.arange(critical_steps_start, critical_steps_end + 1)

def compute_rl_fid(diff, len, diff_max, len_max=200, eps=0.001, weight=1):
    diff = diff / diff_max
    len = len / len_max
    diff_log = np.log(np.mean(diff,1))
    len_log = np.log(np.mean(len, 1))
    return len_log - weight*diff_log

run_rl_fid = True

encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
embed_dim = 3
likelihood_type = 'regression'



# # Explainer 6 - DGP.
path = save_path+'dgp_regression_GRU_600_False_False_False_False_False_False_True_0.01_10_16_True_3_exp.npz'
dgp_fid_results = np.load(path)
dgp_sal = dgp_fid_results['sal']
dgp_fid = dgp_fid_results['fid']
dgp_stab = dgp_fid_results['stab']


dgp_fid = np.vstack((dgp_fid, dgp_stab[None, ...]))

fid_all = np.vstack((dgp_fid[None, ...]))

fid_all = fid_all[:, 1:, :]

explainer_all = ['EDGE']
metrics_all = ['Top10', 'Top20', 'Top30', 'Stability']
save_fig_path = save_path+'model_fid_stab.pdf'


if run_rl_fid:
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from pendulum.utils import rl_fed

    env_name = 'Pendulum-v0'
    max_ep_len = 100



    model = model = PPO.load('agents/Pendulum-v0')
    env = make_vec_env(env_name)

    num_trajs = 500

    # Baseline fidelity
    diff_all_10 = np.zeros((1, num_trajs))
    diff_all_20 = np.zeros((1, num_trajs))
    diff_all_30 = np.zeros((1, num_trajs))
    diff_all_40 = np.zeros((1, num_trajs))

    importance_len_10 = np.zeros((1, num_trajs))
    importance_len_20 = np.zeros((1, num_trajs))
    importance_len_30 = np.zeros((1, num_trajs))
    importance_len_40 = np.zeros((1, num_trajs))
    finals_all = np.zeros(num_trajs)
    exps_all = [dgp_sal]

    loader = trange(num_trajs, desc='Loading data')

    for k,importance in enumerate(exps_all):
        print(k)
        for i in trange(num_trajs, desc=f'exp {k}'):
            value = importance[i,0]

            importance_traj = np.argsort(importance[i,])[::-1]
            importance_traj_10 = select_critical_steps(importance_traj, 0.1)
            importance_traj_20 = select_critical_steps(importance_traj, 0.2)
            importance_traj_30 = select_critical_steps(importance_traj, 0.3)
            importance_traj_40 = select_critical_steps(importance_traj, 0.4)
            original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
            orin_reward = original_traj['final_rewards']

            if k == 0:
                finals_all[i] = orin_reward
            orin_reward = sum(original_traj['rewards'])
            seed = int(original_traj['seed'])+123456
            # rl_fed(env=env, seed=seed, model=model, original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
            #        render=False, mask_act=False)
            replay_reward_10 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_10, render=False, mask_act=True)

            replay_reward_20 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_20, render=False, mask_act=True)

            replay_reward_30 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_30, render=False, mask_act=True)

            replay_reward_40 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_40, render=False, mask_act=True)

            diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
            diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
            diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
            diff_all_40[k, i] = np.abs(orin_reward-replay_reward_40)

            importance_len_10[k, i] = len(importance_traj_10)            
            importance_len_20[k, i] = len(importance_traj_20)
            importance_len_30[k, i] = len(importance_traj_30)
            importance_len_40[k, i] = len(importance_traj_40)

    np.savez(save_path+'fid_rl.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_20=diff_all_20, diff_40=diff_all_40,
             len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, len_40=importance_len_40, rewards=finals_all)


# Reward diff and explanation len figures

diff_10 = np.load('exp_results/fid_rl.npz')['diff_10']
diff_20 = np.load('exp_results/fid_rl.npz')['diff_20']
diff_30 = np.load('exp_results/fid_rl.npz')['diff_30']
diff_40 = np.load('exp_results/fid_rl.npz')['diff_40']

len_10 = np.load('exp_results/fid_rl.npz')['len_10']
len_20 = np.load('exp_results/fid_rl.npz')['len_20']
len_30 = np.load('exp_results/fid_rl.npz')['len_30']
len_40 = np.load('exp_results/fid_rl.npz')['len_40']

eps = 0.0001
rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=diff_10.max(), eps=eps)
rl_fid_20 = compute_rl_fid(diff_20, len_20, diff_max=diff_20.max(), eps=eps)
rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=diff_30.max(), eps=eps)
rl_fid_40 = compute_rl_fid(diff_40, len_40, diff_max=diff_40.max(), eps=eps)

print(rl_fid_10)
print(rl_fid_20)
print(rl_fid_30)
#print(np.std(rl_fid_30, 1))
print(rl_fid_40)
