import sys

import pyspiel
import torch
import numpy as np
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as az_model
from mask_net import MLP
from ppo_gmax import Config, watcher, load_pretrain, _init_bot, Trajectory


EXP_ID = 0


def select_top_k_states(logger, trajectory, states, k):
    importance_scores = trajectory.mask_probs
    ind = np.argsort(importance_scores)[-k:]

    ind_rescale = [2*i for i in ind]
    selected_importance_scores = [importance_scores[int(i)] for i in ind]

    
    logger.print("----Top " + str(k) + " important steps----")
    logger.print(ind_rescale)
    logger.print(selected_importance_scores)




def select_bottom_k_states(logger, trajectory, states, k):
    importance_scores = trajectory.mask_probs
    ind = np.argsort(importance_scores)[:k]
    ind_rescale = [2*i for i in ind]
    selected_importance_scores = [importance_scores[int(i)] for i in ind]
    logger.print("----Top " + str(k) + " least important steps----")
    logger.print(ind_rescale)
    logger.print(selected_importance_scores)

def play_game(logger, game_num, game, bots, mask_net, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  logger.print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.print("Initial state:\n{}".format(state))
  state_buffer = []

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
        mask_action = dist.sample()
        log_prob = dist.log_prob(mask_action).detach().numpy()
        mask_action = mask_action.detach().numpy()
        if mask_action[0] == 0:
          trajectory.mask_pos.append(trajectory.eps_len)
        trajectory.mask_probs.append(log_prob[0])
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    
      state_buffer.append(state)

    trajectory.act_seq.append(action)
    trajectory.eps_len += 1
    logger.print("State " + str(trajectory.eps_len))
    logger.print(state)

 
  trajectory.returns = state.returns()

  if trajectory.mask_probs[0] == trajectory.mask_probs[1]:
    print("Equal")
  
  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  

  select_top_k_states(logger, trajectory, state_buffer, k=5)
  select_bottom_k_states(logger, trajectory, state_buffer, k=5)
  return trajectory

@watcher
def visualize(*, game, config, logger):
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))

  model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
  model.load_checkpoint("/data/zelei/masknet_break300/models/checkpoints/model_300.pth")
  model.eval()
  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()
  # define alphazero and pure mcts
  az_player = EXP_ID

  results = []

  logger.print("--------Start visualizing-------")
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
  for game_num in range(5):
    trajectory = play_game(logger, game_num, game, bots, model, temperature=1,temperature_drop=0)
    results.append(trajectory.returns[az_player])



def run(config: Config):
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

  visualize(game=game, config=config)

