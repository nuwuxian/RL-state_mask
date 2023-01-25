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
mask_agent_path = '/data/jiahao/mujoco/fidelity_kicker/agent-zoo/KickAndDefend-v0_1_MLP_MLP_0_const_-1_const_0_const_False/20220716_180950-0/checkpoints/000019906560/model.pkl'
adv_ismlp = True
mask_obs_normpath = '/data/jiahao/mujoco/fidelity_kicker/agent-zoo/KickAndDefend-v0_1_MLP_MLP_0_const_-1_const_0_const_False/20220716_180950-0/checkpoints/000019906560/obs_rms.pkl'
mask_action_space = spaces.Discrete(2)
# Load agent, build environment, and play an episode.
env = gym.make(env_name)
env.seed(1)

victim_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=2, version=1)
adv_agent = make_zoo_agent(env_name, env.observation_space.spaces[1], env.action_space.spaces[1], tag=1, version=1, scope='adv_agent')
with tf.variable_scope("mask_agent", reuse=False):
    mask_agent = make_adv_agent(env.observation_space.spaces[1], mask_action_space,1,mask_agent_path,adv_ismlp,mask_obs_normpath, name='mask_agent')


traj_path = 'trajs/' + env_name.split('/')[1]
#traj_path = 'trajs/Pong-v0.npz'
num_traj = 500
max_ep_len = 400


if os.path.isdir("recording"):
    os.system("rm -rf recording")


os.system("mkdir recording")

rollout(victim_agent, adv_agent, mask_agent, env, num_traj=num_traj, max_ep_len=max_ep_len, save_path=traj_path)