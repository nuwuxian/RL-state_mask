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

flags.DEFINE_integer("num_tests", 500, "total test rounds")
FLAGS = flags.FLAGS

@watcher
def test(*, game, config, test_idx, steps, logger):
  print("Process ", test_idx)
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))

  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path + str(steps))
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()
  # define alphazero and pure mcts
  az_player = EXP_ID

  results = []
  path_1 = "retrain_model/ours/" 


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

    trajectory = _play_game(logger, game_num, game, bots, None, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
    
    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results)))

  logger.print("Average reward: ", np.mean(results))

  avg_reward_filename = path_1 + "avg_reward_" + str(test_idx) + ".out" 
  np.savetxt(avg_reward_filename, results)

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

  reward_hist = []

  for steps in range(200):

    test_processes = [multiprocessing.Process(target=test, kwargs={"game": game, "config": config, "test_idx": i, "steps": steps + 1, "num": i}) for i in range(int(FLAGS.num_tests/10))]

    for p in test_processes:
      p.start()
    
    for p in test_processes:
      p.join()

    baseline_rewards = []
    for test_idx in range((int(FLAGS.num_tests/10))):
      path = "retrain_model/ours/"  + "avg_reward_" + str(test_idx) +".out"
      baseline_reward = np.loadtxt(path)
      baseline_rewards.append(np.mean(baseline_reward))
      
    baseline_reward = np.mean(baseline_rewards)  
    print("Checkpoint-" + str(steps))
    print("Baseline performance:", baseline_reward)

    reward_hist.append(baseline_reward)
  
  np.savetxt("ours.out", reward_hist)
