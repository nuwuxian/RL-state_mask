import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time
import random
import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

GAMMA = 0.99 
LAM = 0.95

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_buffer(free_queue,
               full_queue,
               buffers,
               flags):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.cat([buffers[key][m] for m in indices], dim=0)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    position = flags.position
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        x_dim = 319 if position == 'landlord' else 430
        specs = dict(
            done=dict(size=(T,), dtype=torch.bool),
            reward=dict(size=(T,), dtype=torch.float32),
            value = dict(size=(T,), dtype=torch.float32),
            logpac = dict(size=(T,), dtype=torch.float32),
            ret = dict(size=(T,), dtype=torch.float32),
            adv = dict(size=(T,), dtype=torch.float32),
            obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
            act=dict(size=(T,), dtype=torch.int8),
            obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                if not device == "cpu":
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                else:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, mask_net, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    exp_id = mask_net.position
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        obs_x_no_action_buf = []
        obs_z_buf = []
        act_buf = []
        value_buf = []
        reward_buf = []
        done_buf = []
        logpac_buf = []
        # ret, adv 
        ret_buf = []
        adv_buf = []
        sz, game_len = 0, 0

        position, obs, env_output = env.initial()
        while True:
            while True:
                if position == exp_id:
                    obs_x_no_action_buf.append(env_output['obs_x_no_action'])
                    obs_z_buf.append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                if position == exp_id and mask_net != None:
                    dist, value = mask_net.inference(env_output['obs_z'], env_output['obs_x_no_action'])
                    mask_action = dist.sample()
                    if mask_action == 0:
                        action = random.choice(obs['legal_actions'])
                    log_prob = dist.log_prob(mask_action)
                    act_buf.append(mask_action.cpu())
                    value_buf.append(value.cpu())
                    logpac_buf.append(log_prob.cpu())
                    # psuedo fill
                    ret_buf.append(0)
                    adv_buf.append(0)
                    sz += 1
                    game_len += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    # exp id
                    diff = sz - len(reward_buf)
                    if diff > 0:
                        done_buf.extend([False for _ in range(diff)])
                        reward = env_output['episode_return'] if exp_id == 'landlord' else -env_output['episode_return']
                        reward_buf.extend([0.0 for _ in range(diff-1)])
                        reward_buf.append(reward)
                    break
            done = True 
            last_values, lastgaelam = 0, 0
            # returns, advs
            for t in reversed(range(sz-game_len, sz)):
                if t == sz - 1:
                   nextnonterminal = 1.0 - done
                   nextvalues = last_values
                else:
                   nextnonterminal = 1.0 - done_buf[t+1]
                   nextvalues = value_buf[t+1]
                delta = reward_buf[t] + GAMMA * nextvalues * nextnonterminal - value_buf[t]
                adv_buf[t] = lastgaelam = delta + GAMMA * LAM * nextnonterminal * lastgaelam
                ret_buf[t] = adv_buf[t] + value_buf[t]
            # reset game length
            game_len = 0
            done_buf[-1] = True
            while sz > T: 
                index = free_queue.get()
                if index is None:
                    break
                for t in range(T):
                    buffers['done'][index][t, ...] = done_buf[t]
                    buffers['reward'][index][t, ...] = reward_buf[t]
                    buffers['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[t]
                    buffers['act'][index][t, ...] = act_buf[t]
                    buffers['value'][index][t, ...] = value_buf[t]
                    buffers['logpac'][index][t, ...] = logpac_buf[t]
                    buffers['obs_z'][index][t, ...] = obs_z_buf[t]
                    buffers['ret'][index][t, ...] = ret_buf[t]
                    buffers['adv'][index][t, ...] = adv_buf[t]
                full_queue.put(index)
                done_buf = done_buf[T:]
                reward_buf = reward_buf[T:]
                obs_x_no_action_buf = obs_x_no_action_buf[T:]
                act_buf = act_buf[T:]
                value_buf = value_buf[T:]
                logpac_buf = logpac_buf[T:]
                obs_z_buf = obs_z_buf[T:]
                ret_buf = ret_buf[T:]
                adv_buf = adv_buf[T:]
                sz -= T
    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix