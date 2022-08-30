import os
import sys
import math
import torch
import pyspiel
import numpy as np
from ppo_gmax import Trajectory, watcher, load_pretrain, _init_bot, _play_game, Config
from mask_net import Conv2d
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import multiprocessing as mp
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from absl import flags

flags.DEFINE_string("masknet_path", "/home/zxc5262/test_conv/models/checkpoints/model_-1.pth", "Where to save masknet checkpoints.")
flags.DEFINE_integer("num_tests", 500, "total test rounds")
FLAGS = flags.FLAGS

def fid_play(logger, game_num, game, bots, mask_net, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  threshold, attack_num = 0.9, 0
  EXP_ID = 0

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
        # return dis is a tensor, value is a scalar
        obs = np.array(state.observation_tensor())
        dist, value = mask_net.inference(obs)
        mask_action = dist.sample().detach().numpy()
        log_prob = dist.log_prob(torch.Tensor([1])).detach().numpy()
         # attack when the mask prob larger than a threshod
        if math.exp(log_prob[0]) > threshold:
          action = np.random.choice(state.legal_actions())
          attack_num += 1

      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    
    trajectory.act_seq.append(action)
    trajectory.eps_len += 1
 
  trajectory.returns = state.returns()

  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  
  ret = 0
  if trajectory.returns[EXP_ID] == 1:
    ret = 1
  return ret, attack_num * 2.0 / trajectory.eps_len


@watcher
def attack(*, game, config, test_idx, logger):
  print("Process ", test_idx)
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))

  model = None
  if config.test_masknet:
     model = Conv2d(input_size, config.nn_width, config.nn_depth, 2, config.path)
     model.load_checkpoint(FLAGS.masknet_path)

  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()

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

    base_win, attack_win = 0, 0
    attack_ratio = []
    for game_num in range(10):

      ret_0, _ = fid_play(logger, game_num, game, bots, None, 
                       temperature=1, temperature_drop=0)
      ret_1, ratio = fid_play(logger, game_num, game, bots, model, 
                       temperature=1, temperature_drop=0)

      base_win += ret_0
      attack_win += ret_1
      attack_ratio.append(ratio)

    # ret the wining rate
    base_win = base_win * 1.0 / 10
    attack_win = attack_win * 1.0 / 10
    attack_ratio = np.mean(attack_ratio)

    # return a tuple
    np.savetxt("base_win_" + str(test_idx) + ".out", [base_win])
    np.savetxt("attack_win_" + str(test_idx) + ".out", [attack_win])
    np.savetxt("attack_ratio_" + str(test_idx) + ".out", [attack_ratio])


def alpha_zero_attack(config):
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


  processes = []

  for i in range(int(FLAGS.num_tests/10)):
    p = mp.Process(
            target=attack,
            kwargs={"game": game, "config": config, "test_idx": i, "num": i},
        )
    p.start()
    processes.append(p)

  for p in processes:
    p.join()

  base_win, attack_win, attack_ratio = 0, 0, 0.0
  for test_idx in range(int(FLAGS.num_tests/10)):

    base_win += np.loadtxt("base_win_" + str(test_idx) + ".out")
    attack_win += np.loadtxt("attack_win_" + str(test_idx) + ".out")
    attack_ratio += np.loadtxt("attack_ratio_" + str(test_idx) + ".out")

  num_threads = int(FLAGS.num_tests/10)

  print("Average winning rate before: ", base_win * 1.0 / num_threads)
  print("Average winning rate after: ", attack_win * 1.0 / num_threads)
  print("Average attack steps: ", attack_ratio * 1.0 / num_threads)
