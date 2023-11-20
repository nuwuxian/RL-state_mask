import os, sys
sys.path.append('..')
import numpy as np
import gym, argparse
import timeit
import gym_compete
import tensorflow as tf
from agent_fidelity import make_zoo_agent,make_adv_agent
from gym import spaces
# Setup env, load the target agent, and collect the trajectories.

def rollout(victim_agent, adv_agent, mask_agent, env, num_traj, max_ep_len=1e3, attack=False, attack_type='our'):
    # load agent-0 / agent-1 
    np.random.seed(100)
    reward_record = []
    win_count = 0
    threshold = 0.9999
    ratio_attack = []
    vic_action_seq = []

    for i in range(num_traj):
        env.seed(i+2000)
        if i % 100 == 0:
            print('Traj %d out of %d.' %(i, num_traj))
        observation = env.reset()
        episode_length, epr, eploss, done = 0, 0, 0, False
        attack_num = 0
        while not done and episode_length < max_ep_len:    
            actions = []
            for id, obs in enumerate(observation):
                if id == 0:
                   act = victim_agent.act(observation=obs)[0]
                   vic_action_seq.append(act)
                   clipped_actions = np.clip(act, env.action_space.spaces[0].low, env.action_space.spaces[0].high)
                else:
                    act, value = adv_agent.act(observation=obs)
                    if attack: 
                        '''mask action'''
                        if attack_type == 'our':
                            mask_act, neglogp = mask_agent.act(observation=obs)
                            prob = np.exp(-neglogp)[0]
                            if mask_act == 1:
                                prob = 1 - prob
                            # print(mask_act,prob)
                        elif attack_type == 'max_value':
                            prob = value['vpred']
                            # print(prob)
                        if prob > threshold:
                            act = act + np.random.rand(act.shape[0]) * 3 - 1
                            attack_num += 1
                    clipped_actions = np.clip(act, env.action_space.spaces[0].low, env.action_space.spaces[0].high)
                actions.append(clipped_actions)

            actions = tuple(actions)
            observation, rewards, dones, infos = env.step(actions)
            done = dones
            episode_length += 1
        ratio_attack.append(attack_num * 1.0 / episode_length)
        if 'winner' in infos[1]:
            win_count += 1
    if attack: 
       ratio_attack = np.mean(ratio_attack)
       return win_count * 1.0 / num_traj, ratio_attack
    else:
       return win_count * 1.0 / num_traj

tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))

env_name = 'multicomp/YouShallNotPassHumans-v0'

mask_agent_path = '/data/jiahao/mujoco/fidelity_runner/agent-zoo/YouShallNotPassHumans-v0_0_MLP_MLP_0_const_-1_const_0_const_False/20220830_162540-0/checkpoints/000019906560/model.pkl'
mask_obs_normpath = '/data/jiahao/mujoco/fidelity_runner/agent-zoo/YouShallNotPassHumans-v0_0_MLP_MLP_0_const_-1_const_0_const_False/20220830_162540-0/checkpoints/000019906560/obs_rms.pkl'
mask_action_space = spaces.Discrete(2)
# Load agent, build environment, and play an episode.
env = gym.make(env_name)
env.seed(1)
x = env.action_space
victim_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=1, version=1)
adv_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=2, version=1, scope='adv_agent')
adv_ismlp = True
with tf.variable_scope("mask_agent", reuse=False):
     mask_agent = make_adv_agent(env.observation_space.spaces[1], mask_action_space, 1, mask_agent_path, adv_ismlp,mask_obs_normpath, name='mask_agent')

num_traj = 500
max_ep_len = 400
# our attack

# win_rate = rollout(victim_agent, adv_agent, mask_agent, env, num_traj=num_traj, max_ep_len=400, attack=False)
win_rate_attack, attack_ratio = rollout(victim_agent, adv_agent, mask_agent, env, num_traj=num_traj, max_ep_len=400, attack_type='our', attack=True)
# print("Average winning rate before: ", win_rate)
print("Average winning rate after: ", win_rate_attack)
print("Average attack ratio: ", attack_ratio)