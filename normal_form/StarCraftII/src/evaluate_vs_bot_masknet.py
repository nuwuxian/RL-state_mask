from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import multiprocessing
from envs.raw_env import SC2RawEnv
from envs.actions.zerg_action_wrappers import ZergActionWrapper
from envs.observations.zerg_observation_wrappers import ZergObservationWrapper
from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent
from agents.ppo_policies import LstmPolicy, MlpPolicy, Mask_MlpPolicy
from agents.ppo_agent import PPOAgent, transform_tuple

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", 500, "Number of episodes to evaluate.")
flags.DEFINE_string("model_path", '/data/jiahao/TencentSC2/StarCraftII/normal-agent/checkpoint-400000', "Filepath to load initial model.")
flags.DEFINE_boolean("disable_fog", True, "Disable fog-of-war.")
flags.DEFINE_enum("difficulty", 'A',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")

flags.DEFINE_enum("agent", 'ppo', ['ppo', 'dqn', 'random', 'keyboard'],
                  "Agent name.")
flags.DEFINE_enum("policy", 'mlp', ['mlp', 'lstm'], "Job type.")
flags.DEFINE_string("game_version", '4.6', "Game core version.")
flags.DEFINE_integer("step_mul", 32, "Game steps per agent step.")
flags.DEFINE_boolean("use_all_combat_actions", False, "Use all combat actions.")
flags.DEFINE_boolean("use_region_features", False, "Use region features")
flags.DEFINE_boolean("use_action_mask", True, "Use action mask or not.")
flags.DEFINE_string("mask_path", '/data/jiahao/TencentSC2/StarCraftII/mask_agent/checkpoint-1150000', "Filepath to load masknet model.")
flags.FLAGS(sys.argv)

SAVE_PATH = '../../results/bot'
GAME_SEED = 1234


def create_env(random_seed=None):
    #  create environment.
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


def evaluate(game_seed):
    env = create_env(game_seed)

    if FLAGS.agent == 'ppo':
        mask_agent = create_ppo_agent(env,policy=Mask_MlpPolicy)
        base_agent = create_ppo_agent(env)
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        win_games = 0
        loss_games = 0
        tie_games = 0
        cum_return = 0.0
        _done = False
        _mask_state = mask_agent._model.initial_state
        _base_state = base_agent._model.initial_state
        for i in range(FLAGS.num_episodes):
              observation = env.reset()
              mask_agent.reset()
              base_agent.reset()
              done, step_id = False, 0
              while not done:
                base_action, _, _base_state, _, mask = base_agent._model.step(
                    transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                    _base_state,
                    np.expand_dims(_done, 0))

                # mask_action, mask_value, _mask_state, mask_neglogpac = mask_agent._model.step(
                #     transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                #     _mask_state,
                #     np.expand_dims(_done, 0))

                # prob = np.exp(-mask_neglogpac)[0]
                # if mask_action[0]==1:
                #     legal_action = np.where(mask[0]==1)[0]
                #     base_action[0] = np.random.choice(legal_action)

                observation, reward, done, _ = env.step(base_action[0])
                cum_return += reward
                step_id += 1
              if done:
                  if reward == 1:
                      win_games += 1
                  elif reward == -1:
                      loss_games += 1
                  else:
                      tie_games += 1
        print("Evaluated %d/%d Episodes Avg Return %f Avg Winning Rate %f Win %d Lose %d tie %d" %
              (i + 1, FLAGS.num_episodes, cum_return / (i + 1),
               ((cum_return / (i + 1)) + 1) / 2.0, win_games, loss_games, tie_games))
        return (((cum_return / (FLAGS.num_episodes)) + 1) / 2.0, cum_return, win_games, loss_games, tie_games)
    except KeyboardInterrupt: pass
    finally: env.close()


def main(argv):
    logging.set_verbosity(logging.ERROR)
    winning_info = evaluate(GAME_SEED)
    model = FLAGS.model_path.split('/')[-1]
    np.save(os.path.join(SAVE_PATH, model+'_'+FLAGS.difficulty), winning_info)


if __name__ == '__main__':
    app.run(main)
