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
from generate_eval_data import generate

def load_card_play_models(card_play_model_path_dict):

    model = Model()
    model.eval()

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model.get_model(position).load_state_dict(torch.load(card_play_model_path_dict[position], map_location='cuda:0'))

    masknet = MaskNet()
    masknet.get_model().load_state_dict(torch.load(card_play_model_path_dict['masknet'], map_location='cuda:0'))
    masknet.eval()
    return model, masknet

def select_steps(path, critical, import_thrd):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(10):
    mask_probs_path = path + "mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)
    
    sorted_idx = np.argsort(confs)

    k = max(int(iteration_ends * import_thrd),1)

    if critical:
    #find the top k:
      idx = sorted_idx[-k:]
 
    else:
    #find the bottom k:
      idx = sorted_idx[:k]

    idx.sort()

    steps_start = idx[0]
    steps_end = idx[0]

    ans = 0
    count = 0

    tmp_end = idx[0]
    tmp_start = idx[0]

    for i in range(1, len(idx)):
      if idx[i] == idx[i - 1] + 1:
        count += 1
        tmp_end = idx[i]
      else:
        count = 1
        tmp_start = idx[i]
        tmp_end = idx[i]
             
      if count > ans:
        ans = count
        steps_start = tmp_start
        steps_end = tmp_end

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


def replay(env, model, step_start, step_end, orig_traj_len, exp_id, act_buf, card_play_data, random_replace=False):

    recorded_actions = act_buf
    game_len = 0
    position, obs, env_output = env.initial(card_play_data)
    
    if random_replace:
        random_replacement_steps = step_end - step_start
        start_range = int(orig_traj_len/3 - random_replacement_steps)
        step_start = np.random.choice(start_range)
        step_end = step_start + random_replacement_steps

    while True:
        if game_len < 3*step_start:
            action = recorded_actions[game_len]

        elif game_len <= 3*step_end:
            print(obs['legal_actions'])
            action = random.choice(obs['legal_actions'])
        
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
            break
    
    return reward

def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios

    fids = []

    for j in range(len(p_ls)):
        p_l = p_ls[j]
        p_d = np.abs(results[j]-replay_results[j])/2
        if p_l == 0 or p_d ==0:
            p_l = 0.001
            p_d = 0.001
        fids.append(np.log(p_l) - np.log(p_d))
  
    return np.mean(fids)

def mp_simulate(card_play_model_path_dict, q, test_idx):
    path = str(test_idx) + "/"
    if not os.path.isdir(path):
        os.system("mkdir " + path)

    objective = 'wp'
    model, masknet = load_card_play_models(card_play_model_path_dict)

    exp_id = masknet.position

    env = Env(objective)
    env = Environment(env, 0)
    reward_buf = []
    game_len_buf = []

    card_play_data = []
    for _ in range(10):
        card_play_data.append(generate())

    for game_num in range(10):
        act_buf = []
        logpac_buf = []

        game_len = 0
        position, obs, env_output = env.initial(card_play_data[game_num])
        while True:
            with torch.no_grad():
                agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs['legal_actions'][_action_idx]
            act_buf.append(action)
            if position == exp_id and masknet != None:
                dist, value = masknet.inference(env_output['obs_z'], env_output['obs_x_no_action'])
                log_prob = dist.log_prob(torch.Tensor([1]).to('cuda:0'))
                logpac_buf.append(log_prob.cpu())
            game_len += 1
            position, obs, env_output = env.step(action)
            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                reward_buf.append(reward)
                game_len_buf.append(game_len)
                break
        
        eps_len_filename = path + "eps_len_" + str(game_num) + ".out" 
        np.savetxt(eps_len_filename, [game_len])

        #act_seq_filename = path + "act_seq_" + str(game_num) + ".out" 
        #np.savetxt(act_seq_filename, act_buf, fmt='%s')

        mask_probs_filename = path + "mask_probs_" + str(game_num) + ".out" 
        np.savetxt(mask_probs_filename, logpac_buf)

    np.savetxt(path + "reward_record.out", reward_buf)
    results = np.loadtxt(path + "reward_record.out")

    baseline_reward = np.mean(reward_buf)
    q.put((baseline_reward))

    important_thresholds=[0.4, 0.3, 0.2, 0.1]

    for i in range(len(important_thresholds)):

        select_steps(path, critical=True, import_thrd=important_thresholds[i])

        critical_steps_starts = np.loadtxt(path + "critical_steps_starts.out")
        critical_steps_ends = np.loadtxt(path + "critical_steps_ends.out")


        critical_ratios = []
        replay_results= []
        for game_num in range(10):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            critical_step_start = critical_steps_starts[game_num]
            critical_step_end = critical_steps_ends[game_num]
            critical_ratios.append(3*(critical_step_end - critical_step_start + 1)/orig_traj_len)
            replay_result = replay(env, model, critical_step_start, critical_step_end, orig_traj_len, exp_id, act_buf,
                                   card_play_data[game_num], random_replace=False)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_reward_record.out",  [np.mean(replay_results)])

        fid_score = cal_fidelity_score(critical_ratios, results, replay_results)
        np.savetxt(path + str(i) + "_fid_score.out", [fid_score])

        replay_results= []
        for game_num in range(10):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            critical_step_start = critical_steps_starts[game_num]
            critical_step_end = critical_steps_ends[game_num]
            replay_result = replay(path, game_num, env, model, critical_step_start, critical_step_end, orig_traj_len, exp_id, act_buf,
                                   card_play_data[game_num], random_replace=True)
            replay_results.append(replay_result)


        np.savetxt(path + str(i) + "_replay_rand_reward_record.out", np.mean(replay_results))

        select_steps(path, critical=False, import_thrd=important_thresholds[i])

        noncritical_steps_starts = np.loadtxt(path + "non_critical_steps_starts.out")
        noncritical_steps_ends = np.loadtxt(path + "non_critical_steps_ends.out")

  
        replay_results= []
        for game_num in range(10):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            noncritical_step_start = noncritical_steps_starts[game_num]
            noncritical_step_end = noncritical_steps_ends[game_num]
            replay_result = replay(path, game_num, env, model, noncritical_step_start, noncritical_step_end, orig_traj_len, exp_id, act_buf,
                                   card_play_data[game_num], random_replace=False)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_non_reward_record.out",  [np.mean(replay_results)])


        replay_results= []
        for game_num in range(10):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            noncritical_step_start = noncritical_steps_starts[game_num]
            noncritical_step_end = noncritical_steps_ends[game_num]
            replay_result = replay(path, game_num, env, model, noncritical_step_start, noncritical_step_end, orig_traj_len, exp_id, act_buf,
                                   card_play_data[game_num], random_replace=True)
            replay_results.append(replay_result)

        np.savetxt(path + str(i) + "_replay_rand_non_reward_record.out", np.mean(replay_results))



def evaluate(landlord, landlord_up, landlord_down, masknet, num_workers):


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
 
    for i in range(num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_model_path_dict, q, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    critical_performs = [[],[],[],[]]
    fidelity_scores = [[],[],[],[]]
    rand_critical_performs = [[],[],[],[]]
    noncritical_performs = [[],[],[],[]]
    rand_noncritical_performs = [[],[],[],[]]

    for i in range(num_workers):
        result = q.get()
        rewards.append(result) 

        path = str(i) + "/"

        for j in range(4):
            critical_perform = np.loadtxt(path + str(j) + "_replay_reward_record.out")
            critical_performs[j].append(critical_perform)
            fidelity_score = np.loadtxt(path +str(j) + "_fid_score.out")
            fidelity_scores[j].append(fidelity_score)
            rand_critical_perform = np.loadtxt(path + str(j) + "_replay_rand_reward_record.out")
            rand_critical_performs[j].append(rand_critical_perform)
            noncritical_perform = np.loadtxt(path + str(j) + "_replay_non_reward_record.out")
            noncritical_performs[j].append(noncritical_perform)
            rand_noncritical_perform = np.loadtxt(path + str(j) + "_replay_non_rand_reward_record.out")
            rand_noncritical_performs[j].append(rand_noncritical_perform)
    
    critical_perform = []
    fidelity_score = []
    rand_critical_perform = []
    noncritical_perform = []
    rand_noncritical_perform = []

    for i in range(4):
        critical_perform.append(np.mean(critical_performs[i]))
        fidelity_score.append(np.mean(fidelity_scores[i]))
        rand_critical_perform.append(np.mean(rand_critical_performs[i]))
        noncritical_perform.append(np.mean(noncritical_performs[i]))
        rand_noncritical_perform.append(np.mean(rand_noncritical_performs[i]))


    print('WP results (baseline):')
    print('landlord : {}'.format(np.mean(rewards)))
    print("Replay (important): ", critical_perform)
    print("Fidelity score: ", fidelity_score)
    print("Replay (rand important): ", rand_critical_perform)
    print("Replay (nonimportant): ", noncritical_perform)
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
            default='douzero_wp_checkpoints/douzero/landlord_masknet_weights_21533400.ckpt')
    parser.add_argument('--num_workers', type=int, default=1)
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
             args.num_workers)