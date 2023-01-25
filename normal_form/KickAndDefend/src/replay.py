import os, sys
sys.path.append('..')
import numpy as np
import gym, argparse
import timeit
import gym_compete
from rollout import rollout
import tensorflow as tf
from agent_fidelity import make_zoo_agent,make_adv_agent
from gym import spaces
# Setup env, load the target agent, and collect the trajectories.


tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1)
sess = tf.Session(config=tf_config)
sess.__enter__()
sess.run(tf.variables_initializer(tf.global_variables()))


env_name = 'multicomp/KickAndDefend-v0'
mask_action_space = spaces.Discrete(2)
# Load agent, build environment, and play an episode.
env = gym.make(env_name)
env.seed(1)

victim_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=2, version=1)
adv_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=1, version=1, scope='adv_agent')


num_traj = 500
max_ep_len = 400
valid_num = num_traj
replay_rewards = []
# np.random.seed(100)
critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")
win_count = 0

original_rewards = np.loadtxt("./recording/reward_record.out")

for i in range(num_traj):
    if i % 50 == 0:
        print("Process traj:", i)
    if original_rewards[i]==0:
        valid_num -= 1
        replay_rewards.append(0)
        continue
    vic_action_sequence_path = "./recording/vic_act_seq_" + str(i) + ".out"
    vic_recorded_actions = np.loadtxt(vic_action_sequence_path)

    adv_action_sequence_path = "./recording/adv_act_seq_" + str(i) + ".out"
    adv_recorded_actions = np.loadtxt(adv_action_sequence_path)

    env.seed(i)
    observation = env.reset()
    victim_agent.reset()
    adv_agent.reset()
    episode_length, epr, eploss, _done = 0, 0, 0, False  # bookkeeping
    
    while not _done and episode_length < max_ep_len:
        
        actions = []
        if episode_length < critical_steps_starts[i]:
            clip_adv_action = adv_recorded_actions[episode_length]
            clip_vic_action = vic_recorded_actions[episode_length]
            actions.append(clip_adv_action)
            actions.append(clip_vic_action)
        
        elif episode_length <= critical_steps_ends[i]:
            for id, obs in enumerate(observation):
                if id==1:
                   act, _ = victim_agent.act(observation=obs)
                   clipped_actions = act
                else:
                    act, _ = adv_agent.act(observation=obs)
                    act = act + np.random.rand(act.shape[0]) * 3 -1
                    clipped_actions = np.clip(act, -0.4, 0.4)
                actions.append(clipped_actions)

        else:
            for id, obs in enumerate(observation):
                if id==1:
                   act, _ = victim_agent.act(observation=obs)
                   clipped_actions = act
                else:
                    act, _ = adv_agent.act(observation=obs)
                    '''mask action'''
                    # mask_act, _ = mask_agent.act(observation=obs)
                    # mask_act = mask_act[0]
                    # if mask_act == 1:
                    #     act = act + np.random.rand(act.shape[0]) * 3 -1
                    '''mask action finish'''
                    clipped_actions = act
                actions.append(clipped_actions)


        actions = tuple(actions)
        observation, rewards, done, infos = env.step(actions)
        _done = done[0]
        episode_length += 1


    # if infos[1]['reward_remaining'] < infos[0]['reward_remaining']:
    #     replay_rewards.append(1)
    #     win_count += 1
    # elif infos[0]['reward_remaining'] == infos[1]['reward_remaining']:
    #     replay_rewards.append(0)
    #     valid_num -= 1
    # else:
    #     replay_rewards.append(-1)

    if 'winner' in infos[0]:
        replay_rewards.append(1)
        win_count += 1
    elif 'winner' in infos[1]:
        replay_rewards.append(-1)
    else:
        replay_rewards.append(0)
        valid_num -= 1

np.savetxt("./recording/replay_reward_record.out", replay_rewards)
print("The winning rate is: ", win_count/ valid_num)