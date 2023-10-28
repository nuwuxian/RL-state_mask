import os
import sys
import torch
import pyspiel
import numpy as np
from absl import flags
from ppo_gmax import Trajectory, watcher, load_pretrain, _init_bot, _play_game, Config
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import multiprocessing

EXP_ID = 0

flags.DEFINE_string("retrain_model_path", "/home/zxc5262/Retrain/TicTacToe/retrain_model/masknet/checkpoint-100", "Where to save masknet checkpoints.")
flags.DEFINE_integer("num_tests", 500, "total test rounds")
FLAGS = flags.FLAGS

    


def fid_play(logger, game_num, game, bots, temperature, temperature_drop):
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

      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    
    trajectory.act_seq.append(action)
    trajectory.eps_len += 1
 
  trajectory.returns = state.returns()

  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory



@watcher
def evaluate(*, game, config, test_idx, logger):
  input_size = int(np.prod(config.observation_shape))
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()
  # define alphazero and pure mcts
  az_player = EXP_ID

  results = []

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
  for game_num in range(500):

    trajectory = fid_play(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
  

  logger.print("Average reward (baseline): ", np.mean(results))


  retrain_model = load_pretrain(FLAGS.retrain_model_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, retrain_model)

  results = []

  logger.print("Testing retrained model")

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
  for game_num in range(500):

    trajectory = fid_play(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
  
  logger.print("Average reward (retrained model): ", np.mean(results))





def alpha_zero_test(config: Config):

  game = pyspiel.load_game(config.game)
  config = config._replace(
    observation_shape=game.observation_tensor_shape(),
    output_size=game.num_distinct_actions())

  print("Starting game", config.game)

  path = config.path
  test_processes = [multiprocessing.Process(target=evaluate, kwargs={"game": game, "config": config, "test_idx": i, "num": i}) for i in range(int(FLAGS.num_tests/500))]

  for p in test_processes:
    p.start()
  
  for p in test_processes:
    p.join()

  