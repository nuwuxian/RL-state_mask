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

def select_steps(path, import_thrd, total_games):

    critical_steps_starts = []
    critical_steps_ends = []


    for i_episode in range(total_games):
        mask_probs_path = path + "mask_probs_" + str(i_episode) + ".out"
        mask_probs = np.loadtxt(mask_probs_path)

        confs = mask_probs

        iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
        iteration_ends = np.loadtxt(iteration_ends_path)

        sorted_idx = np.argsort(confs)

        k = max(int(len(mask_probs) * import_thrd),1)
        idx = sorted_idx[-k:] 
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
                if tmp_prob >= sum_prob:
                    steps_start, steps_end = tmp_start, tmp_end

        critical_steps_starts.append(steps_start)
        critical_steps_ends.append(steps_end)

      
    np.savetxt(path + "critical_steps_starts.out", critical_steps_starts)
    np.savetxt(path + "critical_steps_ends.out", critical_steps_ends)


def mp_simulate(card_play_model_path_dict, q, test_idx, path, game_per_worker):

    objective = 'wp'
    model, masknet = load_card_play_models(card_play_model_path_dict)

    exp_id = masknet.position

    env = Env(objective)
    env = Environment(env, 0)
    reward_buf = []

    winning_games = []
    losing_games = []

    card_play_data_buff = []
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
            print("Finish game ", test_idx * game_per_worker + game_num)
            if reward == 1:
                winning_games.append(test_idx * game_per_worker + game_num)
            else:
                losing_games.append(test_idx * game_per_worker + game_num)
            card_play_data = card_play_data_buff[i]
            eps_len_filename = path + "eps_len_" + str(test_idx * game_per_worker + game_num) + ".out"
            np.savetxt(eps_len_filename, [game_len])

            mask_probs_filename = path + "mask_probs_" + str(test_idx * game_per_worker + game_num) + ".out" 
            np.savetxt(mask_probs_filename, logpac_buf)
            
            act_seq_filename = path + "act_seq_" + str(test_idx * game_per_worker + game_num) + ".npy"
            np.save(act_seq_filename, act_buf, allow_pickle=True)
            
            card_data_filename = path + "card_" + str(test_idx * game_per_worker + game_num) + ".npy"
            np.save(card_data_filename, card_play_data, allow_pickle=True)

            game_num += 1
            if game_num >= game_per_worker:
                break

    assert game_num == game_per_worker

    winning_games_filename = path + str(test_idx) + "_winning_games.out"
    np.savetxt(winning_games_filename, winning_games)

    losing_games_filename = path + str(test_idx) + "_losing_games.out"
    np.savetxt(losing_games_filename, losing_games)


def evaluate(landlord, landlord_up, landlord_down, masknet, total_games, num_workers):
    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down,
        'masknet': masknet}

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    pre_folder = './retrain_data/'
    game_per_worker = int(total_games / num_workers)
 
    for i in range(num_workers):
        p = ctx.Process(
                target=mp_simulate,
                args=(card_play_model_path_dict, q, i, pre_folder, game_per_worker))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    select_steps(pre_folder, 0.3, total_games)

    winning_games_buf = []
    losing_games_buf = []
    for i in range(num_workers):
        winning_games = np.loadtxt(pre_folder +str(i) + "_winning_games.out")
        winning_games_buf.extend(winning_games)
        losing_games = np.loadtxt(pre_folder + str(i) + "_losing_games.out")
        losing_games_buf.extend(losing_games)
        

    np.savetxt(pre_folder+"winning_games.out", winning_games_buf)
    np.savetxt(pre_folder+"losing_games.out", losing_games_buf)


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
            default='landlord_lasso_0.06_batch_42/LR_0.0003_NUM_EPOCH_4_NMINIBATCHES_4/douzero/landlord_masknet_weights_2545200.ckpt')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--total_games', type=int, default=20000)
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






