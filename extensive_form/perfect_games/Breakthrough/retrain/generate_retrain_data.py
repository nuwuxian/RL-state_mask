import os
import sys
import torch
import pyspiel
import numpy as np
from absl import flags
from ppo_gmax import Trajectory, load_pretrain, _init_bot, _play_game, Config
from mask_net import MLP
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import multiprocessing

EXP_ID = 0

flags.DEFINE_string("masknet_path", "masknet/models/checkpoints/model_-1.pth", "Where to save masknet checkpoints.")
flags.DEFINE_string("kdd_path", "masknet/models/checkpoints/model_300.pth", "Where to save masknet checkpoints.")
flags.DEFINE_integer("num_tests", 50000, "total test rounds")
FLAGS = flags.FLAGS

    


def fid_play(game_num, game, bots, mask_net, kdd, temperature, temperature_drop):
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
      policy = np.zeros(game.num_distinct_actions())
      for c in root.children:
        policy[c.action] = c.explore_count
      policy = policy**(1 / temperature)
      policy /= policy.sum()
      if len(actions) >= temperature_drop:
        action = root.best_child().action
      else:
        action = np.random.choice(len(policy), p=policy)
      # add masknet
      if state.current_player() == EXP_ID and mask_net != None:
        value = bots[state.current_player()].evaluator.evaluate(state)[EXP_ID]
        trajectory.values.append(value)

        # return dis is a tensor, value is a scalar
        obs = np.array(state.observation_tensor())
        dist, value = mask_net.inference(obs)
        mask_action = dist.sample()
        log_prob = dist.log_prob(torch.Tensor([1])).detach().numpy()
        mask_action = mask_action.detach().numpy()
        if mask_action[0] == 0:
          trajectory.mask_pos.append(trajectory.eps_len)
        trajectory.mask_probs.append(np.exp(log_prob[0]))

        dist, value = kdd.inference(obs)
        log_prob = dist.log_prob(torch.Tensor([1])).detach().numpy()
        trajectory.kdd_probs.append(np.exp(log_prob[0]))

      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    
    trajectory.act_seq.append(action)
    trajectory.eps_len += 1
 
  trajectory.returns = state.returns()

  print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory

def select_steps(path, critical = True, import_thrd = 0.3):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(50000):
    mask_probs_path = path + "mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)
    
    sorted_idx = np.argsort(confs)
    
    k = max(int(iteration_ends * import_thrd),1)

    if critical:
    #find the top k:
      idx = sorted_idx[-k:]
 
    else:
    #find the bottom k:
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




def collect(game, config, test_idx):
  print("Process ", test_idx)

  input_size = int(np.prod(config.observation_shape))

  model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
  model.load_checkpoint(FLAGS.masknet_path)
  model.eval()

  kdd_model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
  kdd_model.load_checkpoint(FLAGS.kdd_path)
  kdd_model.eval()

  print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()
  # define alphazero and pure mcts
  az_player = EXP_ID

  results = []

  path = config.path
  cmd_1 = "mkdir " + path

  if not os.path.isdir(path):
    os.system(cmd_1)


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

  print("---start collecting trajs for retrain---")

  winning_results = []
  losing_results = []
  for game_num in range(1000):

      trajectory = fid_play(game_num, game, bots, model, kdd_model, temperature=1,
                            temperature_drop=0)
      result = trajectory.returns[az_player]
      if result == 1:
        winning_results.append(test_idx*1000 + game_num)
      else:
        losing_results.append(test_idx*1000 + game_num)
    
      mask_probs = trajectory.mask_probs
      mask_pos = trajectory.mask_pos
      action_seq = trajectory.act_seq
      eps_len = trajectory.eps_len
      values = trajectory.values
      kdd_probs = trajectory.kdd_probs

      mask_pos_filename = path + "mask_pos_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(mask_pos_filename, mask_pos)

      eps_len_filename = path + "eps_len_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(eps_len_filename, [eps_len])

      act_seq_filename = path + "act_seq_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(act_seq_filename, action_seq)

      mask_probs_filename = path + "mask_probs_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(mask_probs_filename, mask_probs)

      values_filename = path + "values_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(values_filename, values)

      kdd_probs_filename = path + "kdd_probs_" + str(test_idx*1000 + game_num) + ".out" 
      np.savetxt(kdd_probs_filename, kdd_probs)


  np.savetxt(path + "winning_games_" + str(test_idx) + ".out", winning_results)
  np.savetxt(path + "losing_games_" + str(test_idx) + ".out", losing_results)





def generate_retrain_data(config: Config):

  game = pyspiel.load_game(config.game)
  config = config._replace(
    observation_shape=game.observation_tensor_shape(),
    output_size=game.num_distinct_actions())
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  print("Starting game", config.game)

  path = config.path
  collect_processes = [multiprocessing.Process(target=collect, kwargs={"game": game, "config": config, "test_idx": i}) for i in range(int(FLAGS.num_tests/1000))]

  for p in collect_processes:
    p.start()
  
  for p in collect_processes:
    p.join()

  select_steps(config.path)
