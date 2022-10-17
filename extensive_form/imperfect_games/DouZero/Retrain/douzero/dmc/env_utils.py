"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 

class Training_pool():
    def __init__(self, losing_games_file, winning_games_file, ratio):
        self.total_num = 5000
        self.ratio = ratio
        self.losing_games_idxs = self.extract_idxs(losing_games_file)
        self.winning_games_idxs = self.extract_idxs(winning_games_file)
        self.candidates = self.create_pool()
    
    def extract_idxs(self, filename):
        idxs = np.loadtxt(filename)
        return idxs
    
    def create_pool(self):
        losing_idxs_selected = np.random.choice(self.losing_games_idxs, int(self.total_num * self.ratio))
        winning_idxs_selected = np.random.choice(self.winning_games_idxs, int(self.total_num * (1-self.ratio)))
        pool = np.concatenate((losing_idxs_selected, winning_idxs_selected), axis=None)
        return pool

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None
        self.exp_id = 'landlord'

        # training pool 
        losing_game_file = '/data/zelei/DouZero_lasso_0.06/retrain_data/losing_games.out'
        winning_games_file = '/data/zelei/DouZero_lasso_0.06/retrain_data/winning_games.out'

        ratio = 0.8
        train_pool = Training_pool(losing_game_file, winning_games_file, ratio)
        self.idxs_list = train_pool.candidates
        self.critical_steps_starts = np.loadtxt('/data/zelei/DouZero_lasso_0.06/retrain_data/critical_steps_starts.out')

    def reset(self):
        path = '/data/zelei/DouZero_lasso_0.06/retrain_data/'
        idx = int(np.random.choice(self.idxs_list))
        cards = np.load(path + "card_" + str(idx) + ".npy", allow_pickle=True)[()]
        print(cards)
        recorded_actions = np.load(path + "act_seq_" + str(idx) + ".npy", allow_pickle=True)
        step_start = self.critical_steps_starts[idx]
        position, obs, env_output = self.initial(cards)
        game_len, count = 0, 0
        
        while count < step_start:
            action = recorded_actions[game_len]
            if position == self.exp_id:
                count += 1
            
            position, obs, env_output = self.step(action)
            game_len += 1
        
        return position, obs, env_output

    def initial(self, cards=None):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(cards), self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )

    def close(self):
        self.env.close()
