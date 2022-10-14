import os 
import time
import torch
import random
import argparse
import multiprocessing as mp
import numpy as np
from douzero.dmc.models import Model
from douzero.dmc.masknet import MaskNet
from douzero.env.game import GameEnv
from douzero.env.env import Env, get_obs
from douzero.dmc.env_utils import Environment, _format_observation
from generate_eval_data import generate

threshold = 0.2


def load_card_play_models(card_play_model_path_dict):

    model = Model()
    model.eval()

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        model.get_model(position).load_state_dict(torch.load(card_play_model_path_dict[position], map_location='cuda:0'))


    return model



def replay(env, model, exp_id, card_play_data):

    position, obs, env_output = env.initial(card_play_data)

    count, traj_len = 0, 0

    
    while True:
        with torch.no_grad():
            agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
        _action_idx = int(agent_output['action'].cpu().detach().numpy())
        action = obs['legal_actions'][_action_idx]

        if position == exp_id:
            traj_len += 1
            Q_values = model.forward(position, obs['z_batch'], obs['x_batch'], training=True)['values'].detach().cpu().numpy()
            lazy_gap = np.max(Q_values)- np.mean(Q_values)

            if lazy_gap > threshold:
                action = random.choice(obs['legal_actions'])
                count += 1


        position, obs, env_output = env.step(action)
        if env_output['done']:
            utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
            reward = 1 if utility.cpu().numpy() > 0 else 0
            break
    return reward, count/traj_len



def mp_simulate(card_play_model_path_dict, q, test_idx):
    path = str(test_idx) + "/"
    if not os.path.isdir(path):
        os.system("mkdir " + path)

    objective = 'wp'
    model = load_card_play_models(card_play_model_path_dict)

    exp_id = 'landlord'

    env = Env(objective)
    env = Environment(env, 0)
    reward_buf = []
    game_len_buf = []

    card_play_data = []
    for _ in range(25):
        card_play_data.append(generate())

    for game_num in range(25):
        obs_buf = []
        act_buf = []
        values = []
        game_len = 0
        position, obs, env_output = env.initial(card_play_data[game_num])
        while True:
            with torch.no_grad():
                agent_output = model.forward(position, obs['z_batch'], obs['x_batch'])
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs['legal_actions'][_action_idx]

            if position == exp_id:
                Q_values = model.forward(position, obs['z_batch'], obs['x_batch'], training=True)['values'].detach().cpu().numpy()
                value = np.max(Q_values)


            game_len += 1
            position, obs, env_output = env.step(action)
            if env_output['done']:
                utility = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                reward = 1 if utility.cpu().numpy() > 0 else 0
                reward_buf.append(reward)
                break


    baseline_reward = np.mean(reward_buf)

    attack_rewards = []
    attack_ratios = []

    for game_num in range(25):
        reward, attack_ratio = replay(env, model, exp_id, card_play_data[game_num])
        attack_rewards.append(reward)
        attack_ratios.append(attack_ratio)
    
    attack_reward = np.mean(attack_rewards)
    avg_attack_ratio = np.mean(attack_ratios)

    q.put((baseline_reward, attack_reward, avg_attack_ratio))



def evaluate(landlord, landlord_up, landlord_down, num_workers):

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

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

    baseline_rewards = []
    attack_rewards = []
    attack_ratios = []

    for i in range(num_workers):
        result = q.get()
        baseline_rewards.append(result[0]) 
        attack_rewards.append(result[1])
        attack_ratios.append(result[2])
    
    print('Before attack : {}'.format(np.mean(baseline_rewards)))
    print('After attack : {}'.format(np.mean(attack_rewards)))
    print('Attack ratio: {}'.format(np.mean(attack_ratios)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_WP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/douzero_WP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/douzero_WP/landlord_down.ckpt')
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--gpu_device', type=str, default='2')
    parser.add_argument('--position', default='landlord', type=str,
                    help='explain position')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device


    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.num_workers)