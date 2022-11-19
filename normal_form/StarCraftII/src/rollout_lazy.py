from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import multiprocessing
from agents.ppo_policies import LstmPolicy, MlpPolicy, Mask_MlpPolicy
from agents.ppo_agent import PPOAgent, transform_tuple

from envs.raw_env import SC2RawEnv
from envs.actions.zerg_action_wrappers import ZergActionWrapper
from envs.observations.zerg_observation_wrappers import ZergObservationWrapper
import copy


from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_integer("num_episodes", 50, "Number of episodes to evaluate.")
flags.DEFINE_enum("difficulty", 'A',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("model_path", "/data/jiahao/TencentSC2/StarCraftII/normal-agent/checkpoint-400000", "Filepath to load initial model.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_boolean("mask_victim", False, "Mask out the part of the victim observation that represents the adversarial")

flags.DEFINE_enum("agent", 'ppo', ['ppo', 'dqn', 'random', 'keyboard'],
                  "Agent name.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_enum("value", 'mlp', ['mlp', 'lstm'], "Value type")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_integer("game_steps_per_episode", 103200, "Maximum steps per episode.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use action mask or not.")
flags.DEFINE_string("mask_path", '/data/jiahao/TencentSC2/StarCraftII/mask_agent/checkpoint-1150000', "Filepath to load masknet model.")
flags.DEFINE_string('mode','rollout', 'model type')
flags.DEFINE_integer('index', 0, 'index of the run')
flags.FLAGS(sys.argv)


def create_env(random_seed=None):
    env = SC2RawEnv(map_name='Flat64',
                    step_mul=FLAGS.step_mul,
                    agent_race='zerg',
                    bot_race='zerg',
                    difficulty=FLAGS.difficulty,
                    disable_fog=FLAGS.disable_fog,
                    random_seed=random_seed)

     # wrap agent action.
    env = ZergActionWrapper(env,
                            game_version=FLAGS.game_version,
                            mask=FLAGS.use_action_mask,
                            use_all_combat_actions=FLAGS.use_all_combat_actions)
    # wrap observation.
    env = ZergObservationWrapper(
        env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features)
    return env


def create_ppo_agent(env, policy=None):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                      intra_op_parallelism_threads=ncpu,
                      inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    # define policy network type.
    if policy==None:
        policy = {'lstm': LstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    # define the ppo agent.
        agent = PPOAgent(env=env, policy=policy, model_path=FLAGS.model_path, scope_name='model')
    else:
        agent = PPOAgent(env=env, policy=policy, model_path=FLAGS.mask_path, scope_name='mask_model')
    return agent


def evaluate(path, save_dir, index):
    env = create_env(1234)
    G_GAE = 0.99
    if FLAGS.agent == 'ppo':
        mask_agent = create_ppo_agent(env,policy=Mask_MlpPolicy)
        agent = create_ppo_agent(env)
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError
    try:
        for i in range(FLAGS.num_episodes):
            _base_state = agent._model.initial_state
            _mask_state = mask_agent._model.initial_state
            iteration_ends_path = "./results_masknet/eps_len_" + str(i) + ".out"
            iteration_ends = np.loadtxt(iteration_ends_path)
            value_seq = []
            for pointer in range(int(iteration_ends)):
                print("pointer: {}, iteration_ends: {}".format(pointer, iteration_ends))
                env.seed(i)
                np.random.seed(i)
                cum_return = 0.0
                step_id = 0
                _done = False
                observation = env.reset()
                agent.reset()
                mask_agent.reset()
                done, step_id = False, 0
                while step_id < pointer:
                    base_action, base_value, _base_state, _, mask = agent._model.step(
                            transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                            _base_state,
                            np.expand_dims(_done, 0))
                    if step_id == pointer-1:
                        legal_action = np.where(mask[0]==1)[0]
                        np.random.seed(index+i+pointer)
                        base_action[0] = np.random.choice(legal_action)
                    if not done:
                        observation, reward, done, _ = env.step(base_action[0])

                    if step_id == pointer-1:
                        if not done:
                            q = agent._model.value(
                                transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                                _base_state,
                                np.expand_dims(_done, 0))
                        else:
                            q = -100000
                        value_seq.append(reward + G_GAE * q)

                    step_id += 1



            if i%1==0:
                print('episode is : ', i, )
            value_seq_filename = './lazy_recording/' + str(index) + '/' + str(i) + '.out'
            np.savetxt(value_seq_filename, value_seq)
            
    except KeyboardInterrupt: pass
    finally: env.close()

def main(argv):
    logging.set_verbosity(logging.ERROR)
    path = 'results/'
    save_dir = 'lazy_recording/' + str(FLAGS.index)
    if not os.path.isdir(path):
        os.system("mkdir " + path)

    if not os.path.isdir(save_dir):
        os.system("mkdir " + save_dir)
    
    evaluate(path, save_dir, index=FLAGS.index)

if __name__ == '__main__':
    app.run(main)
