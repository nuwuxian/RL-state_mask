import os
import sys
import torch
import pyspiel
import numpy as np
from absl import flags
from ppo_gmax import Trajectory, watcher, load_pretrain, _init_bot, _play_game, Config
from mask_net import MLP
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import multiprocessing

EXP_ID = 0

flags.DEFINE_string("masknet_path", "/home/zxc5262/masknet_break300/models/checkpoints/model_300.pth", "Where to save masknet checkpoints.")
flags.DEFINE_integer("num_tests", 500, "total test rounds")
FLAGS = flags.FLAGS

    


def fid_play(logger, game_num, game, bots, mask_net, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()

  while not state.is_terminal():
    if state.is_chance_node():
      # For chance nodes, rollout according to chance node's probability
      # distribution
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = random_state.choice(action_list, p=prob_list)
      state.apply_action(action)
    else:
      root = bots[state.current_player()].mcts_search(state)
      action = root.best_child().action

      # add masknet
      if state.current_player() == EXP_ID and mask_net != None:
        # return dis is a tensor, value is a scalar
        obs = np.array(state.observation_tensor())
        dist, value = mask_net.inference(obs)
        mask_action = dist.sample().detach().numpy()
        log_prob = dist.log_prob(torch.Tensor([1])).detach().numpy()

        if mask_action[0] == 0:
          trajectory.mask_pos.append(trajectory.eps_len)
        trajectory.mask_probs.append(log_prob[0])
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    
    trajectory.act_seq.append(action)
    trajectory.eps_len += 1
 
  trajectory.returns = state.returns()

  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory

def select_steps(path, critical, import_thrd):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(10):
    mask_probs_path = path + "mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)
    
    sorted_idx = np.argsort(confs)
    
    #critical_range = sorted_idx[int(iteration_ends/4):]
    #noncritical_range = sorted_idx[:int(iteration_ends/4)]


    
    k = max(int(iteration_ends * import_thrd),1)

    if critical:
    #find the top k:
      #idx = np.random.choice(critical_range, k)
      idx = sorted_idx[-k:]
 
    else:
    #find the bottom k:
      #idx = np.random.choice(noncritical_range, k)
      idx = sorted_idx[:k]


    idx.sort()


    steps_start = idx[0]
    steps_end = idx[0]

    ans = 0
    count = 0

    tmp_end = idx[0]
    tmp_start = idx[0]

    for i in range(1, len(idx)):
     
      # Check if the current element is
      # equal to previous element +1
      if idx[i] == idx[i - 1] + 1:
        count += 1
        tmp_end = idx[i]
             
      # Reset the count
      else:
        count = 0
        tmp_start = idx[i]
        tmp_end = idx[i]
             
        # Update the maximum
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


def replay(logger, game_num, game, bots, path, step_start, step_end, random_replace, orig_traj_len, temperature, temperature_drop):
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.opt_print("Initial state:\n{}".format(state))

  action_sequence_path = path + "act_seq_" + str(game_num) + ".out"
  recorded_actions = np.loadtxt(action_sequence_path)

  if random_replace:
    random_replacement_steps = step_end - step_start

    start_range = int(orig_traj_len/2 - random_replacement_steps)

    step_start = np.random.choice(start_range)

    step_end = step_start + random_replacement_steps


  while not state.is_terminal():
    if state.is_chance_node():
      # For chance nodes, rollout according to chance node's probability
      # distribution
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = random_state.choice(action_list, p=prob_list)
      state.apply_action(action)
    else:
      if trajectory.eps_len < 2*step_start:
        action = int(recorded_actions[trajectory.eps_len])

      elif trajectory.eps_len <= 2*step_end:
        if state.current_player() == 1:
          root = bots[state.current_player()].mcts_search(state)
          action = root.best_child().action
        else:
          action = np.random.choice(state.legal_actions())

      else:
        root = bots[state.current_player()].mcts_search(state)
        action = root.best_child().action
      
      state.apply_action(action)    
    
    trajectory.eps_len += 1
  
  trajectory.returns = state.returns()
    
  logger.opt_print("Next state:\n{}".format(state))
  logger.print("Game {}: Returns: {};".format(
      game_num, " ".join(map(str, trajectory.returns))))
  return trajectory

def cal_fidelity_score(critical_ratios, results, replay_results):
  p_ls = critical_ratios

  p_ds = []

  for j in range(len(p_ls)):
    p_ds.append(np.abs(results[j]-replay_results[j])/2)
  reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001
  fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
  
  return fid_score




@watcher
def test(*, game, config, test_idx, logger):
  print("Process ", test_idx)
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))

  model = None
  if config.test_masknet:
     model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
     model.load_checkpoint(FLAGS.masknet_path)

  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()
  # define alphazero and pure mcts
  az_player = EXP_ID

  results = []
  if config.test_masknet:
    logger.print("Testing masknet")
    path_1 = config.path + str(test_idx) + "/"
    path_2 = path_1 + "recording/"
    cmd_1 = "mkdir " + path_1
    cmd_2 = "mkdir " + path_2
    if not os.path.isdir(path_1):
      os.system(cmd_1)
    if not os.path.isdir(path_2):
      os.system(cmd_2)

  else:
    logger.print("Testing baseline")

  bots = [
    _init_bot(config, game, az_evaluator, True),
    mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      random_evaluator,
      solve=True,
      verbose=False,
      dont_return_chance_node=True)
  ]
  for game_num in range(10):

    trajectory = _play_game(logger, game_num, game, bots, model, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
  
    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results)))

  logger.print("Average reward: ", np.mean(results))

  avg_reward_filename = path_2 + "avg_reward.out" 
  np.savetxt(avg_reward_filename, results)
 


  if config.test_masknet:
    logger.print("---start collecting trajs for fid test---")
    bots = [
      _init_bot(config, game, az_evaluator, True),
      mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        random_evaluator,
        solve=True,
        verbose=False,
        dont_return_chance_node=True)
    ]

    results=[]
    for game_num in range(10):

      trajectory = fid_play(logger, game_num, game, bots, model, temperature=1,
                            temperature_drop=0)
      results.append(trajectory.returns[az_player])
    
      mask_probs = trajectory.mask_probs
      mask_pos = trajectory.mask_pos
      action_seq = trajectory.act_seq
      eps_len = trajectory.eps_len

      mask_pos_filename = path_2 + "mask_pos_" + str(game_num) + ".out" 
      np.savetxt(mask_pos_filename, mask_pos)

      eps_len_filename = path_2 + "eps_len_" + str(game_num) + ".out" 
      np.savetxt(eps_len_filename, [eps_len])

      act_seq_filename = path_2 + "act_seq_" + str(game_num) + ".out" 
      np.savetxt(act_seq_filename, action_seq)

      mask_probs_filename = path_2 + "mask_probs_" + str(game_num) + ".out" 
      np.savetxt(mask_probs_filename, mask_probs)
  

    np.savetxt(path_2 + "reward_record.out", results)
    results = np.loadtxt(path_2 + "reward_record.out")


    important_thresholds=[0.2, 0.15, 0.1, 0.05]

    for i in range(len(important_thresholds)):
      logger.print("current important threshold: ", important_thresholds[i])

      select_steps(path_2, critical=True, import_thrd=important_thresholds[i])
      
      critical_steps_starts = np.loadtxt(path_2 + "critical_steps_starts.out")
      critical_steps_ends = np.loadtxt(path_2 + "critical_steps_ends.out")

      logger.print("Replay(important)")
      critical_ratios = []
      replay_results= []
      for game_num in range(10):
        orig_traj_len = np.loadtxt(path_2 + "eps_len_"+ str(game_num) + ".out")
        critical_step_start = critical_steps_starts[game_num]
        critical_step_end = critical_steps_ends[game_num]
        critical_ratios.append(2*(critical_step_end - critical_step_start + 1)/orig_traj_len)
 
        trajectory = replay(logger, game_num, game, bots, path_2, critical_step_start, critical_step_end, \
                            random_replace=False, orig_traj_len=orig_traj_len, temperature=1,
                            temperature_drop=0)
        replay_results.append(trajectory.returns[az_player])

      logger.print("Average winning rate: ", np.mean(replay_results))
      np.savetxt(path_2 + str(i) + "_replay_reward_record.out", replay_results)


      fid_score = cal_fidelity_score(critical_ratios, results, replay_results)
      logger.print("fidelity_score: ", fid_score)
      np.savetxt(path_2 + str(i) + "_fid_score.out", [fid_score])


      logger.print("Replay(rand important)")
      replay_results= []
      for game_num in range(10):
        orig_traj_len = np.loadtxt(path_2 + "eps_len_"+ str(game_num) + ".out")
        critical_step_start = critical_steps_starts[game_num]
        critical_step_end = critical_steps_ends[game_num]
        for _ in range(5):
          trajectory = replay(logger, game_num, game, bots, path_2, critical_step_start, critical_step_end, \
                              random_replace=True, orig_traj_len=orig_traj_len, temperature=1,
                              temperature_drop=0)
          replay_results.append(trajectory.returns[az_player])
 
      logger.print("Current average winning rate: ", np.mean(replay_results))
      np.savetxt(path_2 + str(i) + "_replay_rand_reward_record.out", replay_results)


      select_steps(path_2, critical=False, import_thrd=important_thresholds[i])
      
      non_critical_steps_starts = np.loadtxt(path_2 + "non_critical_steps_starts.out")
      non_critical_steps_ends = np.loadtxt(path_2 + "non_critical_steps_ends.out")

      logger.print("Replay(nonimportant)")
      replay_results= []
      for game_num in range(10):
        orig_traj_len = np.loadtxt(path_2 + "eps_len_"+ str(game_num) + ".out")
        non_critical_step_start = non_critical_steps_starts[game_num]
        non_critical_step_end = non_critical_steps_ends[game_num]
        trajectory = replay(logger, game_num, game, bots, path_2, non_critical_step_start, non_critical_step_end, \
                            random_replace=False, orig_traj_len=orig_traj_len, temperature=1,
                              temperature_drop=0)
        replay_results.append(trajectory.returns[az_player])

      logger.print("Average winning rate: ", np.mean(replay_results))
      np.savetxt(path_2 + str(i) + "_replay_non_reward_record.out", replay_results)


      logger.print("Replay(rand nonimportant)")
      replay_results= []
      for game_num in range(10):
        orig_traj_len = np.loadtxt(path_2 + "eps_len_"+ str(game_num) + ".out")
        non_critical_step_start = non_critical_steps_starts[game_num]
        non_critical_step_end = non_critical_steps_ends[game_num]
        for _ in range(5):
          trajectory = replay(logger, game_num, game, bots, path_2, non_critical_step_start, non_critical_step_end, \
                              random_replace=True, orig_traj_len=orig_traj_len, temperature=1,
                              temperature_drop=0)
          replay_results.append(trajectory.returns[az_player])
 
      logger.print("Current average winning rate: ", np.mean(replay_results))
      np.savetxt(path_2 + str(i) + "_replay_non_rand_reward_record.out", replay_results)


def alpha_zero_test(config: Config):

  game = pyspiel.load_game(config.game)
  config = config._replace(
    observation_shape=game.observation_tensor_shape(),
    output_size=game.num_distinct_actions())
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Starting game", config.game)
  if game.num_players() != 2:
    sys.exit("AlphaZero can only handle 2-player games.")
  game_type = game.get_type()
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    raise ValueError("Game must have terminal rewards.")
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Game must have sequential turns.")
  if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    raise ValueError("Game must be deterministic.")
  path = config.path
  test_processes = [multiprocessing.Process(target=test, kwargs={"game": game, "config": config, "test_idx": i, "num": i}) for i in range(int(FLAGS.num_tests/10))]

  for p in test_processes:
    p.start()
  
  for p in test_processes:
    p.join()

  baseline_rewards = []
  avg_rewards = []
  critical_performs = [[],[],[],[]]
  fidelity_scores = [[],[],[],[]]
  rand_critical_performs = [[],[],[],[]]
  noncritical_performs = [[],[],[],[]]
  rand_noncritical_performs = [[],[],[],[]]
  for test_idx in range((int(FLAGS.num_tests/10))):
    path = config.path + str(test_idx) + "/" + "recording/"
    avg_reward = np.loadtxt(path + "avg_reward.out" )
    avg_rewards.append(np.mean(avg_reward))
    baseline_reward = np.loadtxt(path + "reward_record.out")
    baseline_rewards.append(np.mean(baseline_reward))
    for i in range(4):
      critical_perform = np.loadtxt(path + str(i) + "_replay_reward_record.out")
      critical_performs[i].append(np.mean(critical_perform))
      fidelity_score = np.loadtxt(path +str(i) + "_fid_score.out")
      fidelity_scores[i].append(fidelity_score)
      rand_critical_perform = np.loadtxt(path + str(i) + "_replay_rand_reward_record.out")
      rand_critical_performs[i].append(np.mean(rand_critical_perform))
      noncritical_perform = np.loadtxt(path + str(i) + "_replay_non_reward_record.out")
      noncritical_performs[i].append(np.mean(noncritical_perform))
      rand_noncritical_perform = np.loadtxt(path + str(i) + "_replay_non_rand_reward_record.out")
      rand_noncritical_performs[i].append(np.mean(rand_noncritical_perform))
    
  baseline_reward = np.mean(baseline_rewards)  
  avg_reward = np.mean(avg_rewards)
  critical_perform = []
  fidelity_score = []
  rand_critical_perform = []
  noncritical_perform = []
  rand_noncritical_perform = []
  for i in range(4):
    critical_perform.append(np.mean(critical_performs[i]))
    fidelity_score.append(np.mean(fidelity_scores[i]))
    rand_critical_perform.append(np.mean(rand_critical_performs[i]))
    noncritical_perform.append(np.mean(noncritical_performs[i]))
    rand_noncritical_perform.append(np.mean(rand_noncritical_performs[i]))


  print("Baseline performance:", baseline_reward)
  print("Masknet performance: ", avg_reward)
  print("Replay (important): ", critical_perform)
  print("Fidelity score: ", fidelity_score)
  print("Replay (rand important): ", rand_critical_perform)
  print("Replay (nonimportant): ", noncritical_perform)
  print("Replay (rand nonimportant): ", rand_noncritical_perform)
