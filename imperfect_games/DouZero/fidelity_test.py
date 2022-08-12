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
from douzero.env.env import Env
from douzero.dmc.env_utils import Environment

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

  for i_episode in range(100):
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

    for game_num in range(100):
        act_buf = []
        logpac_buf = []

        game_len = 0

        position, obs, env_output = env.initial()
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

        act_seq_filename = path + "act_seq_" + str(game_num) + ".out" 
        np.savetxt(act_seq_filename, act_buf)

        mask_probs_filename = path + "mask_probs_" + str(game_num) + ".out" 
        np.savetxt(mask_probs_filename, logpac_buf)

    np.savetxt(path + "reward_record.out", reward_buf)
    results = np.loadtxt(path + "reward_record.out")

    baseline_reward = np.mean(reward_buf)
    q.put((baseline_reward))

    important_thresholds=[0.4, 0.3, 0.2, 0.1]

    for i in range(len(important_thresholds)):
        print("current important threshold: ", important_thresholds[i])

        select_steps(path, critical=True, import_thrd=important_thresholds[i])



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

    
    for i in range(num_workers):
        result = q.get()
        rewards.append(result) 


    print('WP results (baseline):')
    print('landlord : {}'.format(np.mean(rewards)))

    

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
            default='douzero_checkpoints/douzero/landlord_masknet_weights_4200.ckpt')
    parser.add_argument('--num_workers', type=int, default=5)
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
    