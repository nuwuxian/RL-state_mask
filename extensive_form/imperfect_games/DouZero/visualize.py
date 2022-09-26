import os 
import math
import torch
import random
import argparse
import multiprocessing as mp
import numpy as np
import pickle
import pandas as pd
from douzero.dmc.models import Model
from douzero.dmc.masknet import MaskNet
from douzero.dmc.utils import _cards2tensor
from douzero.env.game import GameEnv
from douzero.env.env import Env, get_obs, _cards2array
from douzero.dmc.env_utils import Environment, _format_observation
from generate_eval_data import generate

Card2Str = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
               11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'Small Joker', 30: 'Large Joker'}


def load_card_play_models(card_play_model_path_dict):
    model = Model()
    model.eval()
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model.get_model(position).load_state_dict(torch.load(card_play_model_path_dict[position], map_location='cuda:0'))

    masknet = MaskNet(position='landlord')
    masknet.get_model().load_state_dict(torch.load(card_play_model_path_dict['masknet'], map_location='cuda:0'))
    masknet.eval()
    return model, masknet

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
    for _ in range(1):
        card_play_data_buff.append(generate())
    game_num = 0
    for i in range(1):
        obs_buf = []
        act_buf = []
        logpac_buf = []
        game_len = 0
        position, obs, env_output = env.initial(card_play_data_buff[i])
        while True:
            infoset= env.env.infoset
            current_player = infoset.player_position
            print("Current_player:")
            print(current_player)
            C = (pd.Series(infoset.player_hand_cards)).map(Card2Str)
            my_handcards = list(C)
            with torch.no_grad():
                agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs['legal_actions'][_action_idx]
            if len(action) == 0:
                action_cards = ['Pass']
            else:
                C = (pd.Series(action)).map(Card2Str)
                action_cards = list(C)

            print("Current handcard:")
            print(my_handcards)
            obs_buf.append(my_handcards)
            print("Action:")
            print(action_cards)
            act_buf.append(action_cards)
            
            if position == exp_id and masknet != None:
                x = torch.cat((env_output['obs_x_no_action'], _cards2tensor(action).to("cuda:0"))).float()
                dist, value = masknet.inference(env_output['obs_z'], x)
                log_prob = dist.log_prob(torch.Tensor([1]).to('cuda:0'))
                prob = math.exp(log_prob.cpu())
                print("Importance score:")
                print(prob)
                logpac_buf.append(prob)
            game_len += 1
            position, obs, env_output = env.step(action)
            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                print("Final reward:")
                print(reward)
                break
        if game_len >= 0:
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
    pre_folder = './vis'
    game_per_worker = int(total_games / num_workers)
 
    for i in range(num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_model_path_dict, q, pre_folder + '/' + str(i), game_per_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    



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
            default='/home/xkw5132/Masknet_explanation/extensive_form/imperfect_games/DouZero/landlord_entropy_0.0/LR_0.0003_NUM_EPOCH_4_NMINIBATCHES_4/douzero/landlord_masknet_weights_9563400.ckpt')
    parser.add_argument('--total_games', type=int, default=1)
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
             args.total_games,
             args.num_workers)