from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import multiprocessing
from agents.ppo_policies import LstmPolicy, MlpPolicy
from agents.ppo_agent import PPOAgent, transform_tuple

from envs.selfplay_raw_env import SC2SelfplayRawEnv
from envs.actions.zerg_action_wrappers import ZergPlayerActionWrapper
from envs.observations.zerg_observation_wrappers \
    import ZergPlayerObservationWrapper


from utils.utils import print_arguments, print_actions, print_action_distribution
from agents.random_agent import RandomAgent
from agents.keyboard_agent import KeyboardAgent


FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_integer("num_episodes", 2, "Number of episodes to evaluate.")
flags.DEFINE_enum("difficulty", '1',
                  ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A'],
                  "Bot's strength.")
flags.DEFINE_string("model_path", "/data/jiahao/TencentSC2/StarCraftII/normal-agent/checkpoint-400000", "Filepath to load initial model.")
flags.DEFINE_string("victim_path", "/data/jiahao/TencentSC2/StarCraftII/normal-agent/checkpoint-400000", "victim_path")
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
flags.FLAGS(sys.argv)



def select_steps(path, critical, import_thrd):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(FLAGS.num_episodes):
    values_path = path + "value_" + str(i_episode) + ".out" 
    values = np.loadtxt(values_path)

    confs = values

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    sorted_idx = np.argsort(confs)

    k = max(int(len(values) * import_thrd),1)
    idx = sorted_idx[-k:] if critical else sorted_idx[:k]
    idx.sort()

    steps_start, steps_end = idx[0], idx[0]
    ans, count = 0, 0
    tmp_end, tmp_start = idx[0], idx[0]

    for i in range(1, len(idx)):
      if idx[i] == idx[i - 1] + 1:
        count += 1
        tmp_end = idx[i]
      else:
        count = 0
        tmp_start = idx[i]
        tmp_end = idx[i]
      if count > ans:
        ans = count
        steps_start = tmp_start
        steps_end = tmp_end

    if critical:
      critical_steps_starts.append(steps_start)
      critical_steps_ends.append(steps_end)

    else:
      non_critical_steps_starts.append(steps_start)
      non_critical_steps_ends.append(steps_end)
      
  if critical:
    np.savetxt(path + "critical_steps_starts.out", critical_steps_starts)
    np.savetxt(path + "critical_steps_ends.out", critical_steps_ends)
  else:
    np.savetxt(path + "non_critical_steps_starts.out", non_critical_steps_starts)
    np.savetxt(path + "non_critical_steps_ends.out", non_critical_steps_ends)

def create_env(random_seed=None):
    env = SC2SelfplayRawEnv(map_name='Flat64',
                            step_mul=FLAGS.step_mul,
                            resolution=16,
                            agent_race='zerg',
                            opponent_race='zerg',
                            tie_to_lose=False,
                            disable_fog=FLAGS.disable_fog,
                            game_steps_per_episode=FLAGS.game_steps_per_episode,
                            random_seed=random_seed)

    env = ZergPlayerActionWrapper(
        player=0,
        env=env,
        game_version=FLAGS.game_version,
        mask=FLAGS.use_action_mask,
        use_all_combat_actions=FLAGS.use_all_combat_actions)
    env = ZergPlayerObservationWrapper(
        player=0,
        env=env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features)

    env = ZergPlayerActionWrapper(
        player=1,
        env=env,
        game_version=FLAGS.game_version,
        mask=FLAGS.use_action_mask,
        use_all_combat_actions=FLAGS.use_all_combat_actions)

    env = ZergPlayerObservationWrapper(
        player=1,
        env=env,
        use_spatial_features=False,
        use_game_progress=(not FLAGS.policy == 'lstm'),
        action_seq_len=1 if FLAGS.policy == 'lstm' else 8,
        use_regions=FLAGS.use_region_features, 
        mask_opponent=FLAGS.mask_victim)

    print(env.observation_space, env.action_space)
    return env


def create_ppo_agent(env, model_path, scope_name):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                      intra_op_parallelism_threads=ncpu,
                      inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()
    # define policy network type.
    policy = {'lstm': LstmPolicy, 'mlp': MlpPolicy}[FLAGS.policy]
    # define the ppo agent.
    agent = PPOAgent(env=env, policy=policy, scope_name=scope_name, model_path=model_path)
    return agent


def evaluate(game_seed, path):
    env = create_env(game_seed)

    if FLAGS.agent == 'ppo':
        agent = create_ppo_agent(env, FLAGS.model_path, "base_model")
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        oppo_agent = create_ppo_agent(env, FLAGS.victim_path, "opponent_model")
        cum_return = 0.0
        act_buf = []
        oppo_act_buf = []
        values = []
        action_counts = [0] * env.action_space.n # number of possible actions.

        observation_0, observation_1 = env.reset()
        agent.reset()
        oppo_agent.reset()
        done, step_id = False, 0
        _state = agent._model.initial_state
        while not done:
            action, value, _, _, _ = agent._model.step(transform_tuple(observation_0, lambda x: np.expand_dims(x, 0)), True)
            action_0 = action[0]
            value_0 = value[0]
            act_buf.append(action_0)
            values.append(value_0)
            action_1, _ = oppo_agent.act(observation_1)
            oppo_act_buf.append(action_1)
            (observation_0, observation_1), reward, done, _ = env.step([action_0, action_1])
            action_counts[action_0] += 1
            cum_return += reward
            step_id += 1
        
        final_return = (cum_return + 1) / 2.0

        eps_len_filename = path + "eps_len_" + str(game_seed) + ".out" 
        np.savetxt(eps_len_filename, [step_id])

        value_filename = path + "value_" + str(game_seed) + ".out" 
        np.savetxt(value_filename, values)
        
        act_seq_filename = path + "act_seq_" + str(game_seed) + ".out"
        np.savetxt(act_seq_filename, act_buf)

        oppo_act_seq_filename = path + "oppo_act_seq_" + str(game_seed) + ".out"
        np.savetxt(oppo_act_seq_filename, oppo_act_buf)

        return final_return 
    except KeyboardInterrupt: pass
    finally: env.close()


def replay(game_seed, step_start, step_end, orig_traj_len, act_buf, oppo_act_buf, random_replace):
    env = create_env(game_seed)

    if FLAGS.agent == 'ppo':
        agent = create_ppo_agent(env, FLAGS.model_path, "model")
    elif FLAGS.agent == 'random':
        agent = RandomAgent(action_space=env.action_space)
    elif FLAGS.agent == 'keyboard':
        agent = KeyboardAgent(action_space=env.action_space)
    else:
        raise NotImplementedError

    try:
        if random_replace:
            random_replacement_steps = step_end - step_start
            start_range = int(np.floor(orig_traj_len - random_replacement_steps))
            step_start = np.random.choice(start_range)
            step_end = step_start + random_replacement_steps

        oppo_agent = create_ppo_agent(env, FLAGS.victim_path, "opponent_model")
        cum_return = 0.0
        action_counts = [0] * env.action_space.n # number of possible actions.

        observation_0, observation_1 = env.reset()
        agent.reset()
        oppo_agent.reset()
        done, step_id = False, 0

        while not done:
            if step_id < step_start:
                action_0 = act_buf[step_id]
                action_1 = oppo_act_buf[step_id]
            elif step_id <= step_end:
                _, _, _, _, mask = agent._model.step(transform_tuple(observation_0, lambda x: np.expand_dims(x, 0)),True)
                legal_action = np.where(mask[0]==1)[0]
                action_0 = np.random.choice(legal_action)
                action_1, _ = oppo_agent.act(observation_1)
            else:
                action_0, _ = agent.act(observation_0)
                action_1, _ = oppo_agent.act(observation_1)

            (observation_0, observation_1), reward, done, _ = env.step([action_0, action_1])    
            cum_return += reward
            step_id += 1
        
        final_return = (cum_return + 1) / 2.0

        return final_return 
    except KeyboardInterrupt: pass
    finally: env.close()    

def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios
    p_ds = []

    for j in range(len(p_ls)):
        p_ds.append(np.abs(results[j]-replay_results[j]))
    reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001

    fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
    return fid_score

def main(argv):
    logging.set_verbosity(logging.ERROR)
    path = 'results/'
    if not os.path.isdir(path):
        os.system("mkdir " + path)

    baseline_returns = []

    for i in range(FLAGS.num_episodes):
        final_return = evaluate(i, path)
        baseline_returns.append(final_return)
    
    np.savetxt(path + "reward_record.out", baseline_returns)
    print("Baseline:")
    print(np.mean(baseline_returns))

    important_thresholds=[0.4, 0.3, 0.2, 0.1]

    for i in range(len(important_thresholds)):

        select_steps(path, critical=True, import_thrd=important_thresholds[i])

        critical_steps_starts = np.loadtxt(path + "critical_steps_starts.out")
        critical_steps_ends = np.loadtxt(path + "critical_steps_ends.out")

        critical_ratios = []
        replay_results= []
        for game_num in range(FLAGS.num_episodes):
            orig_traj_len = np.loadtxt(path + "eps_len_"+ str(game_num) + ".out")
            act_buf = np.loadtxt(path + "act_seq_" + str(game_num) + ".out")
            oppo_act_buf = np.loadtxt(path + "oppo_act_seq_" + str(game_num) + ".out")
            critical_step_start = critical_steps_starts[game_num]
            critical_step_end = critical_steps_ends[game_num]
            critical_ratios.append((critical_step_end - critical_step_start + 1)/orig_traj_len)
            replay_result = replay(game_num, critical_step_start, critical_step_end, orig_traj_len, act_buf, oppo_act_buf, random_replace=False)
            replay_results.append(replay_result)

        np.savetxt(str(i) + "_replay_reward_record.out",  [np.mean(replay_results)])
        np.savetxt(str(i) + "_avg_critical_ratio.out",  [np.mean(critical_ratios)])

        fid_score = cal_fidelity_score(critical_ratios, baseline_returns, replay_results)
        np.savetxt(str(i) + "_fid_score.out", [fid_score])



if __name__ == '__main__':
    app.run(main)
