# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic AlphaZero implementation.

This implements the AlphaZero training algorithm. It spawns N actors which feed
trajectories into a replay buffer which are consumed by a learner. The learner
generates new weights, saves a checkpoint, and tells the actors to update. There
are also M evaluators running games continuously against a standard MCTS+Solver,
though each at a different difficulty (ie number of simulations for MCTS).

Due to the multi-process nature of this algorithm the logs are written to files,
one per process. The learner logs are also output to stdout. The checkpoints are
also written to the same directory.

Links to relevant articles/papers:
  https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
    access link to the AlphaGo Zero nature paper.
  https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
    has an open access link to the AlphaZero science paper.
"""

import collections
import datetime
import functools
import itertools
import json
import os
import random
import sys
import tempfile
import time
import traceback

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.algorithms.alpha_zero.mask_net import MLP
import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats
# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001

# RL hyper-parameter
EXP_ID = 0
GAMMA = 0.99 
LAM = 0.95
clip_param = 0.2

C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient

lambda_1 = 0.0001 # lasso regularization
lambda_2 = 0.0001 # fused lasso regularization

class Config(collections.namedtuple(
    "Config", [
        "game",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",
        "quiet",
        "az_path",
        "n_epochs",
    ])):
  """A config for the model/experiment."""
  pass


class TrajectoryState(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self, obs, action, log_prob, _return, adv):
    self.obs = obs 
    self.action = action
    self.log_prob = log_prob
    self._return= _return
    self.adv= adv

class Trajectory(object):
  def __init__(self):
    self.states = []
    self.returns = None

  def add(self, obs, action, log_prob, _return, adv):
    self.states.append(TrajectoryState(obs, action, log_prob, _return, adv))

class TrainInput(collections.namedtuple(
    "TrainInput", "obs actions log_probs returns advs")):
  """Inputs for training the Model."""

  @staticmethod
  def stack(train_inputs):
    obs, actions, log_probs, returns, advs = zip(*train_inputs)
    return TrainInput(
        np.array(obs, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(log_probs, dtype=np.float32),
        np.array(returns, dtype=np.float32),
        np.array(advs, dtype=np.float32)
    )


class Losses(collections.namedtuple("Losses", "policy value entropy")):
  """Losses from a training step."""

  @property
  def total(self):
    return self.policy + C_1 * self.value - C_2 * self.entropy

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.entropy)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.entropy + other.entropy)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.entropy / n)

class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)

def load_pretrain(az_path):
    return az_model.Model.from_checkpoint(az_path)

def watcher(fn):
  """A decorator to fn/processes that gives a logger and logs exceptions."""
  @functools.wraps(fn)
  def _watcher(*, config, num=None, **kwargs):
    """Wrap the decorated function."""
    name = fn.__name__
    if num is not None:
      name += "-" + str(num)
    with file_logger.FileLogger(config.path, name, config.quiet) as logger:
      print("{} started".format(name))
      logger.print("{} started".format(name))
      try:
        return fn(config=config, logger=logger, **kwargs)
      except Exception as e:
        logger.print("\n".join([
            "",
            " Exception caught ".center(60, "="),
            traceback.format_exc(),
            "=" * 60,
        ]))
        print("Exception caught in {}: {}".format(name, e))
        raise
      finally:
        logger.print("{} exiting".format(name))
        print("{} exiting".format(name))
  return _watcher


def _init_bot(config, game, evaluator_, evaluation):
  """Initializes a bot."""
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      evaluator_,
      solve=False,
      dirichlet_noise=noise,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=False,
      dont_return_chance_node=True)


def _play_game(logger, game_num, game, bots, mask_net, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.opt_print("Initial state:\n{}".format(state))

  mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logpacs = [],[],[],[],[],[]

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
      if state.current_player() == EXP_ID:
        # return dis is a tensor, value is a scalar
        obs = np.array(state.observation_tensor())
        dist, value = mask_net.inference(obs)
        mask_action = dist.sample()
        log_prob = dist.log_prob(mask_action).detach().numpy()
        mask_action = mask_action.detach().numpy()
        if mask_action[0] == 0:
          action = np.random.choice(len(policy), p=policy)
        mb_obs.append(obs)
        mb_rewards.append(0)
        mb_actions.append(mask_action[0])
        mb_values.append(value)
        mb_dones.append(False)
        mb_logpacs.append(log_prob[0])
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      logger.opt_print("Player {} sampled action: {}".format(
          state.current_player(), action_str))
      state.apply_action(action)
  
  trajectory.returns = state.returns()
  mb_rewards[-1] = state.returns()[EXP_ID]
  done = True

  nsteps = len(mb_actions)
  mb_values = np.asarray(mb_values, dtype=np.float32)
  last_values = 0

  # discount/bootstrap off value fn
  mb_returns = np.zeros_like(mb_values)
  mb_advs = np.zeros_like(mb_rewards)
  lastgaelam = 0
  for t in reversed(range(nsteps)):
      if t == nsteps - 1:
         nextnonterminal = 1.0 - done
         nextvalues = last_values
      else:
         nextnonterminal = 1.0 - mb_dones[t+1]
         nextvalues = mb_values[t+1]
      delta = mb_rewards[t] + GAMMA * nextvalues * nextnonterminal - mb_values[t]
      mb_advs[t] = lastgaelam = delta + GAMMA * LAM * nextnonterminal * lastgaelam
  mb_returns = mb_advs + mb_values
  for t in range(nsteps):
    trajectory.add(mb_obs[t], mb_actions[t], mb_logpacs[t], mb_returns[t], mb_advs[t])
    
  logger.opt_print("Next state:\n{}".format(state))
  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory


def update_checkpoint(logger, queue, model, az_evaluator):
  """Read the queue for a checkpoint to load, or an exit signal."""
  path = None
  while True:  # Get the last message, ignore intermediate ones.
    try:
      path = queue.get_nowait()
    except spawn.Empty:
      break
  if path:
    logger.print("Inference cache:", az_evaluator.cache_info())
    logger.print("Loading checkpoint", path)
    model.load_checkpoint(path)
    az_evaluator.clear_cache()
  elif path is not None:  # Empty string means stop this process.
    return False
  return True

@watcher
def actor(*, config, game, logger, queue):
  """An actor process runner that generates games and returns trajectories."""
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))
  model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  bots = [
      _init_bot(config, game, az_evaluator, False),
      _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return
    queue.put(_play_game(logger, game_num, game, bots, model, config.temperature,
                         config.temperature_drop))


@watcher
def evaluator(*, game, config, logger, queue):
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = Buffer(config.evaluation_window)
  logger.print("Initializing model")
  input_size = int(np.prod(config.observation_shape))
  model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path)
  logger.print("Initializing bots")
  base_model = load_pretrain(config.az_path)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, base_model)
  random_evaluator = mcts.RandomRolloutEvaluator()

  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return

    az_player = EXP_ID
    difficulty = (game_num // 2) % config.eval_levels
    max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
    bots = [
        _init_bot(config, game, az_evaluator, True),
        mcts.MCTSBot(
            game,
            config.uct_c,
            max_simulations,
            random_evaluator,
            solve=True,
            verbose=False,
            dont_return_chance_node=True)
    ]
    trajectory = _play_game(logger, game_num, game, bots, model, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
    queue.put((difficulty, trajectory.returns[az_player]))

    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results.data)))
@watcher
def learner(*, game, config, device, actors, evaluators, broadcast_fn, logger):
  """A learner that consumes the replay buffer and trains the network."""
  logger.also_to_stdout = True
  replay_buffer = Buffer(config.replay_buffer_size)
  learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
  logger.print("Initializing model")

  # build model for masknet
  input_size = int(np.prod(config.observation_shape))
  model = MLP(input_size, config.nn_width, config.nn_depth, 2, config.path).to(device)
  save_path = model.save_checkpoint(0)
  logger.print("Initial checkpoint:", save_path)
  # build optimizer
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

  data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
  evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
  total_trajectories = 0

  def trajectory_generator():
    """Merge all the actor queues into a single generator."""
    while True:
      found = 0
      for actor_process in actors:
        try:
          yield actor_process.queue.get_nowait()
        except spawn.Empty:
          pass
        else:
          found += 1
      if found == 0:
        time.sleep(0.01)  # 10ms

  def collect_trajectories():
    """Collects the trajectories from actors into the replay buffer."""
    num_trajectories = 0
    num_states = 0
    for trajectory in trajectory_generator():
      num_trajectories += 1
      num_states += len(trajectory.states)
      game_lengths.add(len(trajectory.states))
      game_lengths_hist.add(len(trajectory.states))

      p1_outcome = trajectory.returns[0]
      if p1_outcome > 0:
        outcomes.add(0)
      elif p1_outcome < 0:
        outcomes.add(1)
      else:
        outcomes.add(2)
      replay_buffer.extend(
          TrainInput(s.obs, s.action, s.log_prob, s._return, s.adv)
          for s in trajectory.states)

      if num_states >= learn_rate:
        break
    return num_trajectories, num_states

  def learn(step):
    losses = []
    """Sample from the replay buffer, update weights and save a checkpoint."""
    for epoch in range(config.n_epochs):
      for _ in range(len(replay_buffer) // config.train_batch_size):
        data = replay_buffer.sample(config.train_batch_size)
        batch = TrainInput.stack(data)

        # shift data to gpu
        obs = torch.Tensor(batch.obs).to(device)
        actions = torch.Tensor(batch.actions).to(device)
        log_probs = torch.Tensor(batch.log_probs).to(device)
        returns = torch.Tensor(batch.returns).to(device)

        # normalize the advs
        advs = (batch.advs - batch.advs.mean())/(batch.advs.std() + 1e-8)
        advs = torch.Tensor(advs).to(device)

        # ppo update
        dist, value = model(obs)
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - log_probs).exp() # new_prob/old_prob
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advs
        actor_loss = - torch.min(surr1, surr2).mean()
        critic_loss = (returns - value).pow(2).mean()
        entropy = dist.entropy().mean()
        num_masks = torch.sum(actions)

        actions_roll = torch.roll(actions, 1, 0)
        cont = torch.sum(torch.square(actions - actions_roll))

        loss = C_1 * critic_loss + actor_loss - C_2 * entropy + lambda_1 * num_masks + lambda_2 * cont 
        losses.append(Losses(actor_loss.item(), critic_loss.item(), entropy.item()))

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
    # Always save a checkpoint, either for keeping or for loading the weights to
    # the actors. It only allows numbers, so use -1 as "latest".
    save_path = model.save_checkpoint(
        step if step % config.checkpoint_freq == 0 else -1)
    # modify the loss
    losses = sum(losses, Losses(0, 0, 0)) / len(losses)
    logger.print(losses)
    logger.print("Checkpoint saved:", save_path)
    return save_path, losses

  last_time = time.time() - 60
  for step in itertools.count(1):
    
    game_lengths.reset()
    game_lengths_hist.reset()
    outcomes.reset()

    num_trajectories, num_states = collect_trajectories()
    total_trajectories += num_trajectories
    now = time.time()
    seconds = now - last_time
    last_time = now

    logger.print("Step:", step)
    logger.print(
        ("Collected {:5} states from {:3} games, {:.1f} states/s. "
         "{:.1f} states/(s*actor), game length: {:.1f}").format(
             num_states, num_trajectories, num_states / seconds,
             num_states / (config.actors * seconds),
             num_states / num_trajectories))
    logger.print("Buffer size: {}. States seen: {}".format(
        len(replay_buffer), replay_buffer.total_seen))

    save_path, losses = learn(step)

    for eval_process in evaluators:
      while True:
        try:
          difficulty, outcome = eval_process.queue.get_nowait()
          evals[difficulty].append(outcome)
        except spawn.Empty:
          break

    batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
    batch_size_stats.add(1)
    data_log.write({
        "step": step,
        "total_states": replay_buffer.total_seen,
        "states_per_s": num_states / seconds,
        "states_per_s_actor": num_states / (config.actors * seconds),
        "total_trajectories": total_trajectories,
        "trajectories_per_s": num_trajectories / seconds,
        "queue_size": 0,  # Only available in C++.
        "game_length": game_lengths.as_dict,
        "game_length_hist": game_lengths_hist.data,
        "outcomes": outcomes.data,
        "eval": {
            "count": evals[0].total_seen,
            "results": [sum(e.data) / len(e) if e else 0 for e in evals],
        },
        "batch_size": batch_size_stats.as_dict,
        "batch_size_hist": [0, 1],
        "loss": {
            "policy": losses.policy,
            "value": losses.value,
            "entropy": losses.entropy,
            "sum": losses.total,
        },
        "cache": {  # Null stats because it's hard to report between processes.
            "size": 0,
            "max_size": 0,
            "usage": 0,
            "requests": 0,
            "requests_per_s": 0,
            "hits": 0,
            "misses": 0,
            "misses_per_s": 0,
            "hit_rate": 0,
        },
    })
    logger.print()

    if config.max_steps > 0 and step >= config.max_steps:
      break

    broadcast_fn(save_path)

def alpha_zero(config: Config):
  """Start all the worker processes for a full alphazero setup."""
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
  if not path:
    path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
    config = config._replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit("{} isn't a directory".format(path))
  print("Writing logs and checkpoints to:", path)

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

  actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                         "num": i})
            for i in range(config.actors)]
  evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                 "num": i})
                for i in range(config.evaluators)]

  def broadcast(msg):
    for proc in actors + evaluators:
      proc.queue.put(msg)

  try:
    learner(game=game, config=config, device=device, actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators, broadcast_fn=broadcast)
  except (KeyboardInterrupt, EOFError):
    print("Caught a KeyboardInterrupt, stopping early.")
  finally:
    broadcast("")
    # for actor processes to join we have to make sure that their q_in is empty,
    # including backed up items
    for proc in actors:
      while proc.exitcode is None:
        while not proc.queue.empty():
          proc.queue.get_nowait()
        proc.join(JOIN_WAIT_DELAY)
    for proc in evaluators:
      proc.join()