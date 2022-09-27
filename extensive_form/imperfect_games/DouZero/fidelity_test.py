import os 
import time
import torch
import random
import argparse
import multiprocessing as mp
import numpy as np
import pickle
from douzero.dmc.models import Model
from douzero.dmc.masknet import MaskNet
from douzero.env.game import GameEnv
from douzero.env.env import Env, get_obs
from douzero.dmc.env_utils import Environment, _format_observation
from douzero.dmc.utils import _cards2tensor
from generate_eval_data import generate

def load_card_play_models(card_play_model_path_dict):
    model = Model()
    model.eval()
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model.get_model(position).load_state_dict(torch.load(card_play_model_path_dict[position], map_location='cuda:0'))

    masknet = MaskNet(position='landlord')
    masknet.get_model().load_state_dict(torch.load(card_play_model_path_dict['masknet'], map_location='cuda:0'))
    masknet.eval()
    return model, masknet

def select_steps(path, critical, import_thrd, game_per_worker):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(game_per_worker):
    mask_probs_path = path + "mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    sorted_idx = np.argsort(confs)

    k = max(int(len(mask_probs) * import_thrd),1)
    idx = sorted_idx[-k:] if critical else sorted_idx[:k]
    idx.sort()

    steps_start, steps_end = idx[0], idx[0]
    ans, sum_prob, count = 0, 0.0, 0
    tmp_end, tmp_start = idx[0], idx[0]
    # find the longest continous sequence
    for i in range(1, len(idx)):
      if idx[i] == idx[i-1] + 1:
        count += 1
        tmp_end = idx[i]
      else:
        count = 0
        tmp_start = idx[i]
        tmp_end = idx[i]
      if count >= ans:
        ans = count
        steps_start = tmp_start
        steps_end = tmp_end
        for j in range(steps_start, steps_end+1):
            sum_prob += confs[j]
    # If multiple exists, return the maximum scores
    count, tmp_end, tmp_start = 0, idx[0], idx[0]
    for i in range(1, len(idx)):
      if idx[i] == idx[i-1] + 1:
        count += 1
        tmp_end = idx[i]
      else:
        count = 0
        tmp_start, tmp_end = idx[i], idx[i]
      if count == ans:
        ans = count
        tmp_prob = 0.0
        for j in range(steps_start, steps_end+1):
            tmp_prob += confs[j]
        if critical and tmp_prob >= sum_prob:
            steps_start, steps_end = tmp_start, tmp_end
        elif not critical and tmp_prob < sum_prob:
            steps_start, steps_end = tmp_start, tmp_end
    if critical:
      critical_steps_starts.append(steps_start)
      critical_steps_ends.append(steps_end)
    else:
      non_critical_steps_starts.append(steps_start)
      non_critical_steps_ends.append(steps_end)
      
  if critical:
    np.savetxt(path + "critical_steps_starts.out", critical_steps_starts)
    np.savetxt(path + "critical_steps_ends.out", critical_steps_ends)
  else:
    np.savetxt(path + "non_critical_steps_starts.out", non_critical_steps_starts)
    np.savetxt(path + "non_critical_steps_ends.out", non_critical_steps_ends)

def replay(env, model, step_start, step_end, orig_traj_len, exp_id, act_buf, obs_buf, card_play_data, random_replace=False):

    recorded_actions = act_buf
    replay_cnt = 5 if random_replace else 1
    step_start, step_end = int(step_start), int(step_end)
    if random_replace:
        random_replacement_steps = step_end - step_start
        start_range = int(np.floor((orig_traj_len+2)/3) - random_replacement_steps)
    rewards = []
    for i in range(replay_cnt):
        if random_replace:
            step_start = np.random.choice(start_range)
            step_end = step_start + random_replacement_steps
        game_len, count = 0, 0
        position, obs, env_output = env.initial(card_play_data)
        if obs['legal_actions'] != obs_buf[0]['legal_actions']:
            print("state different!")
        while True:
            if count < step_start:
                action = recorded_actions[game_len]
                if obs['legal_actions'] != obs_buf[game_len]['legal_actions']:
                    print("state different!")
                if position == exp_id:
                    count += 1
            else:
                if position == exp_id and count <= step_end:
                    with torch.no_grad():
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    gold_action = obs['legal_actions'][_action_idx]
                    if len(obs['legal_actions']) == 1:
                        action = random.choice(obs['legal_actions'])
                    else:
                        obs['legal_actions'].remove(gold_action)
                        action = random.choice(obs['legal_actions'])
                    count += 1
                else:
                    with torch.no_grad():
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx] 
            position, obs, env_output = env.step(action)
            game_len += 1

            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                rewards.append(reward)
                break
    return np.mean(rewards)

def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios
    p_ds = []
    for j in range(len(p_ls)):
        p_d = np.abs(results[j]-replay_results[j])
        p_ds.append(p_d)
    reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001
    fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
  
    return fid_score

def mp_simulate(card_play_model_path_dict, q, test_idx, game_per_worker):
    path = str(test_idx) + "/"
    if not os.path.isdir(path):
        os.makedirs(path)

    objective = 'wp'
    model, masknet = load_card_play_models(card_play_model_path_dict)

    exp_id = masknet.position

    env = Env(objective)
    env = Environment(env, 0)
    reward_buf = []

    card_play_data_buff, card_play_data = [], []
    for _ in range(1000):
        card_play_data_buff.append(generate())
    game_num = 0
    for i in range(1000):
        obs_buf = []
        act_buf = []
        logpac_buf = []
        game_len = 0
        position, obs, env_output = env.initial(card_play_data_buff[i])
        while True:
            with torch.no_grad():
                agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs['legal_actions'][_action_idx]
            act_buf.append(action)
            obs_buf.append(obs)
            if position == exp_id and masknet != None:
                dist, value = masknet.inference(env_output['obs_z'], env_output['obs_x_no_action'])
                log_prob = dist.log_prob(torch.Tensor([1]).to('cuda:0'))
                logpac_buf.append(log_prob.cpu())
            game_len += 1
            position, obs, env_output = env.step(action)
            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                break
        if game_len >= 10:
            reward_buf.append(reward)
            card_play_data.append(card_play_data_buff[i])
            eps_len_filename = path + "eps_len_" + str(game_num) + ".out"
            np.savetxt(eps_len_filename, [game_len])

            mask_probs_filename = path + "mask_probs_" + str(game_num) + ".out" 
            np.savetxt(mask_probs_filename, logpac_buf)
            
            act_seq_filename = path + "act_seq_" + str(game_num) + ".npy"
            np.save(act_seq_filename, act_buf, allow_pickle=True)
            
            obs_filename = path + "obs_" + str(game_num) + ".npy"
            np.save(obs_filename, obs_buf, allow_pickle=True)

            game_num += 1
            if game_num >= game_per_worker:
                break
    assert game_num == game_per_worker
    np.savetxt(path + "reward_record.out", reward_buf)
    results = np.loadtxt(path + "reward_record.out")

    baseline_reward = np.mean(reward_buf)
    q.put((baseline_reward))

    important_thresholds=[0.4, 0.3, 0.2, 0.1]

    for i in range(len(important_thresholds)):

        select_steps(path, critical=True, import_thrd=important_thresholds[i], game_per_worker=game_per_worker)

        critical_steps_starts = np.loadtxt(path + "critical_steps_starts.out")
        critical_steps_ends = np.loadtxt(path + "critical_steps_ends.out")

        critical_ratios = []
        replay_results= []
        for game_num in range(game_per_worker):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            mask_probs = np.loadtxt(path + "mask_probs_"+ str(game_num) + ".out")
            act_buf = np.load(path + "act_seq_" + str(game_num) + ".npy", allow_pickle=True)
            obs_buf = np.load(path + "obs_" + str(game_num) + ".npy", allow_pickle=True)
            critical_step_start = critical_steps_starts[game_num]
            critical_step_end = critical_steps_ends[game_num]
            critical_ratios.append((critical_step_end - critical_step_start + 1)/len(mask_probs))
            replay_result = replay(env, model, critical_step_start, critical_step_end, orig_traj_len, exp_id, act_buf, obs_buf,
                                   card_play_data[game_num], random_replace=False)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_reward_record.out",  [np.mean(replay_results)])
        np.savetxt(path + str(i) + "_avg_critical_ratio.out",  [np.mean(critical_ratios)])

        fid_score = cal_fidelity_score(critical_ratios, results, replay_results)
        np.savetxt(path + str(i) + "_fid_score.out", [fid_score])

        replay_results= []
        for game_num in range(game_per_worker):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            act_buf = np.load(path + "act_seq_" + str(game_num) + ".npy", allow_pickle=True)
            obs_buf = np.load(path + "obs_" + str(game_num) + ".npy", allow_pickle=True)
            critical_step_start = critical_steps_starts[game_num]
            critical_step_end = critical_steps_ends[game_num]
            replay_result = replay(env, model, critical_step_start, critical_step_end, orig_traj_len, exp_id, act_buf, obs_buf,
                                   card_play_data[game_num], random_replace=True)
            replay_results.append(replay_result)


        np.savetxt(path + str(i) + "_replay_rand_reward_record.out", [np.mean(replay_results)])

        select_steps(path, critical=False, import_thrd=important_thresholds[i], game_per_worker=game_per_worker)

        noncritical_steps_starts = np.loadtxt(path + "non_critical_steps_starts.out")
        noncritical_steps_ends = np.loadtxt(path + "non_critical_steps_ends.out")
        
        noncritical_ratios = []
  
        replay_results= []
        for game_num in range(game_per_worker):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            mask_probs = np.loadtxt(path + "mask_probs_"+ str(game_num) + ".out")
            act_buf = np.load(path + "act_seq_" + str(game_num) + ".npy", allow_pickle=True)
            obs_buf = np.load(path + "obs_" + str(game_num) + ".npy", allow_pickle=True)
            noncritical_step_start = noncritical_steps_starts[game_num]
            noncritical_step_end = noncritical_steps_ends[game_num]
            noncritical_ratios.append((noncritical_step_end - noncritical_step_start + 1)/len(mask_probs))
            replay_result = replay(env, model, noncritical_step_start, noncritical_step_end, orig_traj_len, exp_id, act_buf, obs_buf,
                                   card_play_data[game_num], random_replace=False)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_non_reward_record.out",  [np.mean(replay_results)])
        np.savetxt(path + str(i) + "_avg_noncritical_ratio.out",  [np.mean(noncritical_ratios)])


        replay_results= []
        for game_num in range(game_per_worker):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            act_buf = np.load(path + "act_seq_" + str(game_num) + ".npy", allow_pickle=True)
            obs_buf = np.load(path + "obs_" + str(game_num) + ".npy", allow_pickle=True)
            noncritical_step_start = noncritical_steps_starts[game_num]
            noncritical_step_end = noncritical_steps_ends[game_num]
            replay_result = replay(env, model, noncritical_step_start, noncritical_step_end, orig_traj_len, exp_id, act_buf, obs_buf,
                                   card_play_data[game_num], random_replace=True)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_rand_non_reward_record.out", [np.mean(replay_results)])



def evaluate(landlord, landlord_up, landlord_down, masknet, total_games, num_workers):
    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down,
        'masknet': masknet}

    rewards = []
    game_lens = []

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    pre_folder = './results'
    game_per_worker = int(total_games / num_workers)
 
    for i in range(num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_model_path_dict, q, pre_folder + '/' + str(i), game_per_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    critical_performs = [[],[],[],[]]
    fidelity_scores = [[],[],[],[]]
    rand_critical_performs = [[],[],[],[]]
    noncritical_performs = [[],[],[],[]]
    rand_noncritical_performs = [[],[],[],[]]
    critical_ratios = [[],[],[],[]]
    noncritical_ratios = [[],[],[],[]]
    

    for i in range(num_workers):
        result = q.get()
        rewards.append(result) 

        path = pre_folder + '/' + str(i) + '/'

        for j in range(4):
            critical_perform = np.loadtxt(path + str(j) + "_replay_reward_record.out")
            critical_performs[j].append(critical_perform)
            fidelity_score = np.loadtxt(path +str(j) + "_fid_score.out")
            fidelity_scores[j].append(fidelity_score)
            critical_ratio = np.loadtxt(path + str(j) + "_avg_critical_ratio.out")
            critical_ratios[j].append(critical_ratio)
            rand_critical_perform = np.loadtxt(path + str(j) + "_replay_rand_reward_record.out")
            rand_critical_performs[j].append(rand_critical_perform)
            noncritical_perform = np.loadtxt(path + str(j) + "_replay_non_reward_record.out")
            noncritical_performs[j].append(noncritical_perform)
            noncritical_ratio = np.loadtxt(path + str(j) + "_avg_noncritical_ratio.out")
            noncritical_ratios[j].append(noncritical_ratio)
            rand_noncritical_perform = np.loadtxt(path + str(j) + "_replay_rand_non_reward_record.out")
            rand_noncritical_performs[j].append(rand_noncritical_perform)
    
    critical_perform = []
    fidelity_score = []
    rand_critical_perform = []
    noncritical_perform = []
    rand_noncritical_perform = []
    critical_ratio = []
    noncritical_ratio = []

    for i in range(4):
        critical_perform.append(np.mean(critical_performs[i]))
        fidelity_score.append(np.mean(fidelity_scores[i]))
        rand_critical_perform.append(np.mean(rand_critical_performs[i]))
        noncritical_perform.append(np.mean(noncritical_performs[i]))
        rand_noncritical_perform.append(np.mean(rand_noncritical_performs[i]))
        critical_ratio.append(np.mean(critical_ratios[i]))
        noncritical_ratio.append(np.mean(noncritical_ratios[i]))


    print('WP results (baseline):')
    print('landlord : {}'.format(np.mean(rewards)))
    print("Replay (important): ", critical_perform)
    print("Avg critical length ratio: ", critical_ratio)
    print("Fidelity score: ", fidelity_score)
    print("Replay (rand important): ", rand_critical_perform)
    print("Replay (nonimportant): ", noncritical_perform)
    print("Avg noncritical length ratio: ", noncritical_ratio)
    print("Replay (rand nonimportant): ", rand_noncritical_perform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_WP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/douzero_WP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/douzero_WP/landlord_down.ckpt')
    parser.add_argument('--masknet', type=str, 
            default='landlord_lasso_0.06_batch_42/LR_0.0003_NUM_EPOCH_4_NMINIBATCHES_4/douzero/landlord_masknet_weights_19475400.ckpt')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--total_games', type=int, default=500)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--position', default='landlord', type=str,
                    help='explain position')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.masknet,
             args.total_games,
             args.num_workers)
