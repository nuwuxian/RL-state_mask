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
flags.DEFINE_integer('threshold', 4, 'threshold for the number of actions')
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



def replay_traj(env, mask_agent, agent, game_seed, step_start, step_end, orig_traj_len, act_buf, random_replace):
    try:
        if random_replace:
            random_replacement_steps = step_end - step_start
            start_range = int(np.floor(orig_traj_len - random_replacement_steps))
            step_start = np.random.choice(start_range)
            step_end = step_start + random_replacement_steps

        cum_return = 0.0
        env.seed(game_seed)
        observation = env.reset()
        agent.reset()
        mask_agent.reset()
        done, step_id = False, 0
        _base_state = agent._model.initial_state
        _done = False
        while not done:
            if step_id < step_start:
                action = int(act_buf[step_id])
            elif step_id <= step_end:
                _, _, _base_state, _, mask = agent._model.step(
                        transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
                        _base_state,
                        np.expand_dims(_done, 0))
                legal_action = np.where(mask[0]==1)[0]
                action = np.random.choice(legal_action)
            else:
                action = agent.act(observation)[0]

            observation, reward, done, _ = env.step(action)    
            cum_return += reward
            step_id += 1
        
        final_return = (cum_return + 1) / 2.0

        return final_return 
    except KeyboardInterrupt: pass

def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios
    p_ds = []

    for j in range(len(p_ls)):
        p_ds.append(np.abs(results[j]-replay_results[j]))
    reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001

    fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
    return fid_score

def main(argv):
    path = 'results/'
    critical_steps_starts = np.loadtxt(path + str(FLAGS.threshold) + "critical_steps_starts.out")
    critical_steps_ends = np.loadtxt(path + str(FLAGS.threshold) + "critical_steps_ends.out")

    env = create_env(1234)
    mask_agent = create_ppo_agent(env,policy=Mask_MlpPolicy)
    agent = create_ppo_agent(env)
    replay_reward = []
    sum_return = 0
    for game_num in range(FLAGS.num_episodes):
        orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
        act_buf = np.loadtxt(path + "act_seq_" + str(game_num) + ".out")
        critical_step_start = critical_steps_starts[game_num]
        critical_step_end = critical_steps_ends[game_num]

        replay_result = replay_traj(env, mask_agent, agent, game_num, critical_step_start, critical_step_end, orig_traj_len, act_buf, random_replace=False)
        replay_reward.append(replay_result)
        sum_return += replay_result

    np.savetxt(path + str(FLAGS.threshold) + "critical_reward.out", replay_reward)
    print("Average return: {}".format(sum_return / FLAGS.num_episodes))
    # replay_results= []
    # for game_num in range(FLAGS.num_episodes):
    #     orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
    #     act_buf = np.loadtxt(path + "act_seq_" + str(game_num) + ".out")
    #     critical_step_start = critical_steps_starts[game_num]
    #     critical_step_end = critical_steps_ends[game_num]
    #     replay_result = replay(game_num, critical_step_start, critical_step_end, orig_traj_len, act_buf, random_replace=True)
    #     replay_results.append(replay_result)
    # rand_critical_results.append(np.mean(replay_results))
    # select_steps(path, critical=False, import_thrd=important_thresholds[i])
    # non_critical_steps_starts = np.loadtxt(path + "non_critical_steps_starts.out")
    # non_critical_steps_ends = np.loadtxt(path + "non_critical_steps_ends.out")
    # non_critical_ratios_tmp = []
    # replay_results= []
    # for game_num in range(FLAGS.num_episodes):
    #     orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
    #     act_buf = np.loadtxt(path + "act_seq_" + str(game_num) + ".out")
    #     non_critical_step_start = non_critical_steps_starts[game_num]
    #     non_critical_step_end = non_critical_steps_ends[game_num]
    #     non_critical_ratios_tmp.append((non_critical_step_end - non_critical_step_start + 1)/orig_traj_len)
    #     replay_result = replay(game_num, non_critical_step_start, non_critical_step_end, orig_traj_len, act_buf, random_replace=False)
    #     replay_results.append(replay_result)
    # noncritical_results.append(np.mean(replay_results))
    # noncritical_ratios.append(np.mean(non_critical_ratios_tmp))
    # replay_results= []
    # for game_num in range(FLAGS.num_episodes):
    #     orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
    #     act_buf = np.loadtxt(path + "act_seq_" + str(game_num) + ".out")
    #     non_critical_step_start = non_critical_steps_starts[game_num]
    #     non_critical_step_end = non_critical_steps_ends[game_num]
    #     replay_result = replay(game_num, non_critical_step_start, non_critical_step_end, orig_traj_len, act_buf, random_replace=True)
    #     replay_results.append(replay_result)
    # rand_noncritical_results.append(np.mean(replay_results))

    # print("Replay (important):")
    # print(critical_results)
    # print("Critical ratios:")
    # print(critical_ratios)
    # print("Fidelity scores:")
    # print(fid_scores)
    # print("Replay (rand important):")
    # print(rand_critical_results)
    # print("Replay (nonimportant):")
    # print(noncritical_results)
    # print("Noncritical ratios:")
    # print(noncritical_ratios)
    # print("Replay (rand nonimportant):")
    # print(rand_noncritical_results)



if __name__ == '__main__':
    app.run(main)
