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



from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_integer("num_episodes", 20, "Number of episodes to evaluate.")
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


def evaluate(path, mode='mask_evaluate'):
    env = create_env(1234)
    attack_ratio = []
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
        return_sum = 0
        return_list = []
        for i in range(FLAGS.num_episodes):
            env.seed(i)
            np.random.seed(i)
            num_attack = 0
            cum_return = 0.0
            act_buf = []
            mask_probs = []
            _done = False
            observation = env.reset()
            agent.reset()
            mask_agent.reset()
            done, step_id = False, 0
            _base_state = agent._model.initial_state
            _mask_state = mask_agent._model.initial_state
            while not done:
                base_action, base_value, _base_state, _, mask = agent._model.step(
                        transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                        _base_state,
                        np.expand_dims(_done, 0))

                prob = base_value
                # print(prob)
                if prob>=0.7:
                    # print('mask')
                    legal_action = np.where(mask[0]==1)[0]
                    base_action[0] = np.random.choice(legal_action)
                    num_attack+=1

                act_buf.append(base_action[0])

                observation, reward, done, _ = env.step(base_action[0])
                cum_return += reward
                step_id += 1
            attack_ratio.append(num_attack/step_id)
            print("attack ratio: ", num_attack/step_id)
            final_return = (cum_return + 1) / 2.0
            return_list.append(final_return)
            return_sum += final_return
            if i%20==0:
                print('Episode: {}, Return: {}'.format(i, return_sum/(i+1)))
            if mode=='rollout':
                eps_len_filename = path + "eps_len_" + str(i) + ".out" 
                np.savetxt(eps_len_filename, [step_id])

                mask_probs_filename = path + "mask_probs_" + str(i) + ".out" 
                np.savetxt(mask_probs_filename, mask_probs)

                act_seq_filename = path + "act_seq_" + str(i) + ".out"
                np.savetxt(act_seq_filename, act_buf)
        print('Average Return: {}'.format(return_sum/FLAGS.num_episodes))
        np.savetxt(path + "reward.out", return_list)
        print('Average Attack Ratio: {}'.format(np.mean(attack_ratio)))

    except KeyboardInterrupt: pass
    finally: env.close()

def main(argv):
    logging.set_verbosity(logging.ERROR)
    path = 'results/'
    if not os.path.isdir(path):
        os.system("mkdir " + path)
    
    evaluate(path, mode='mask_evaluate')

if __name__ == '__main__':
    app.run(main)
