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



def mp_simulate(card_play_model_path_dict, q):

    objective = 'wp'
    model, masknet = load_card_play_models(card_play_model_path_dict)

    exp_id = masknet.position

    env = Env(objective)
    env = Environment(env, 0)
    reward_buf = []
    game_len_buf = []

    for i in range(30):
        act_buf = []
        logpac_buf = []
        game_len = 0

        position, obs, env_output = env.initial()
        while True:
            with torch.no_grad():
                agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs['legal_actions'][_action_idx]
            if position == exp_id and masknet != None:
                dist, value = masknet.inference(env_output['obs_z'], env_output['obs_x_no_action'])
                mask_action = dist.sample()
                if mask_action == 0:
                    action = random.choice(obs['legal_actions'])
                log_prob = dist.log_prob(mask_action)
                act_buf.append(mask_action.cpu())
                logpac_buf.append(log_prob.cpu())
                game_len += 1
            position, obs, env_output = env.step(action)
            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                reward_buf.append(reward)
                game_len_buf.append(game_len)
                break

    q.put((np.mean(reward_buf), np.mean(game_len_buf)))

    """
    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
         ))
    """

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
 
    for _ in range(num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    
    for i in range(num_workers):
        result = q.get()
        rewards.append(result[0]) 
        game_lens.append(result[1])


    print('WP results:')
    print('landlord : {}'.format(np.mean(rewards)))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--masknet', type=str, 
            default='douzero_model/douzero/landlord_masknet_weights_20689200.ckpt')
    parser.add_argument('--num_workers', type=int, default=15)
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
    
