import os
import PIL
import gym
import torch
import base64
import imageio
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from model import CNN

ENV_ID = "Pong-v0"
H_SIZE = 256 # hidden size, linear units of the output layer
L_RATE = 1e-6 # learning rate, gradient coefficient for CNN weight update
G_GAE = 0.99 # gamma param for GAE
L_GAE = 0.95 # lambda param for GAE
E_CLIP = 0.2 # clipping coefficient
C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient
lambda_1 = 0.0001 # lasso regularization
N = 8 # simultaneous processing environments
T = 256 # PPO steps
M = 64 # mini batch size
K = 10 # PPO epochs
T_EPOCHS = 10 # each T_EPOCH
N_TESTS = 500 # do N_TESTS tests
TARGET_REWARD = 0.95
TRANSFER_LEARNING = True
BASELINE_PATH = "./ppo_test/baseline/Pong-v0_+0.896_12150.dat"
MASK_PATH = "./ppo_test/masknet/Pong-v0_+0.910_20170.dat"
critical_steps_starts = np.loadtxt("./retrain_data/critical_steps_starts.out")


for func in [
             lambda: os.mkdir(os.path.join('.', 'ppo_test/checkpoints'))
             ]: # create directories
   try:
       func()
   except Exception as error:
       print (error)
       continue





class training_pool():
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
    



def make_env():    # this function creates a single environment
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _thunk():
        env = gym.make(ENV_ID).env
        return env
    return _thunk

def env_reset(i_episode, envs):
    i_episode = int(i_episode)
    action_sequence_path = "./retrain_data/act_seq_" + str(i_episode) + ".out"
    recorded_actions = np.loadtxt(action_sequence_path)

    envs.seed(i_episode)
    state = envs.reset()
    state = grey_crop_resize_batch(state)
    
    count = 0


    while count < critical_steps_starts[i_episode]:
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)

        actions = np.repeat(int(recorded_actions[count]), N)
        next_state, reward, done, _ = envs.step(actions)
        
        count += 1
        

        next_state = grey_crop_resize_batch(next_state)
        state = next_state
    
    return state


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8) # prevent 0 fraction
    return x

def test_env(i_episode, env, model, device):
    env.seed(i_episode+100000)
    state = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0).to(device)

        baseline_dist, _ = model(state)
        #baseline_action = baseline_dist.sample().cpu().numpy()[0]
        baseline_action = np.argmax(baseline_dist.probs.detach().cpu().numpy()[0])


        next_state, reward, done, _ = env.step(baseline_action)
        next_state = grey_crop_resize(next_state)
        state = next_state
        total_reward += reward
        done = reward
    if total_reward == 1:
        return 1
    else:
        return 0



def grey_crop_resize_batch(state):  # deal with batch observations
    states = []
    for i in state:
        img = Image.fromarray(i)
        grey_img = img.convert(mode='L')
        left = 0
        top = 34  # empirically chosen
        right = 160
        bottom = 194  # empiricallly chosen
        cropped_img = grey_img.crop((left, top, right, bottom)) # cropped image of above dimension
        resized_img = cropped_img.resize((84, 84))
        array_2d = np.asarray(resized_img)
        array_3d = np.expand_dims(array_2d, axis=0)
        array_4d = np.expand_dims(array_3d, axis=0)
        states.append(array_4d)
        states_array = np.vstack(states) # turn the stack into array
    return states_array # B*C*H*W

def grey_crop_resize(state): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left = 0
    top = 34  # empirically chosen
    right = 160
    bottom = 194  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d # C*H*W

def compute_gae(next_value, rewards, masks, values, gamma=G_GAE, lam=L_GAE):
    values = values + [next_value] # concat last value to the list
    gae = 0 # first gae always to 0
    returns = []

    for step in reversed(range(len(rewards))): # for each positions with respect to the result of the action
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] # compute delta, sum of current reward and the expected goodness of the next state (next state val minus current state val), zero if 'done' is reached, so i can't consider next val
        gae = delta + gamma * lam * masks[step] * gae # recursively compute the sum of the gae until last state is reached, gae is computed summing all gae of previous actions, higher is multiple good actions succeds, lower otherwhise
        returns.insert(0, gae + values[step]) # sum again the value of current action, so a state is better to state in if next increment as well
    return returns



def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0) # length of data collected
    
    for _ in range(batch_size // M):

        rand_ids = np.random.randint(0, batch_size, M)  # integer array of random indices for selecting M mini batches
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]



def ppo_update(states, actions, log_probs, returns, advantages, clip_param=E_CLIP):

    for _ in range(K):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            action = action.reshape(1, len(action)) # take the relative action and take the column
            new_log_probs = dist.log_prob(action)
            new_log_probs = new_log_probs.reshape(len(old_log_probs), 1) # take the column
            ratio = (new_log_probs - old_log_probs).exp() # new_prob/old_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = C_1 * critic_loss + actor_loss - C_2 * entropy  # loss function clip+vs+f
            optimizer.zero_grad() # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # performs the parameters update based on the current gradient and the update rule



def ppo_train(model, envs, device, use_cuda, test_rewards, test_epochs, train_epoch, best_reward, losing_games_file, winning_games_file, ratio):
    env = gym.make(ENV_ID).env

    train_pool = training_pool(losing_games_file, winning_games_file, ratio)
    idxs_list = train_pool.candidates


    for i in range(len(idxs_list)):
        
        state = env_reset(idxs_list[i], envs)

        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []


        for t in range(T):

            state = torch.FloatTensor(np.copy(state)).to(device)
            
            dist, value = model(state)
            action = dist.sample().cuda() if use_cuda else dist.sample()

            next_state, reward, done, _ = envs.step(action)
            next_state = grey_crop_resize_batch(next_state) # simplify perceptions (grayscale-> crop-> resize) to train CNN
            
            
            log_prob = dist.log_prob(action) # needed to compute probability ratio r(theta) that prevent policy to vary too much probability related to each action (make the computations more robust)
            log_prob_vect = log_prob.reshape(len(log_prob), 1) # transpose from row to column
            log_probs.append(log_prob_vect)
            action_vect = action.reshape(len(action), 1) # transpose from row to column
            actions.append(action_vect)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            states.append(state)
            
            state = next_state



        next_state = torch.FloatTensor(next_state).to(device) # consider last state of the collection step
        _, next_value = model(next_state) # collect last value effect of the last collection step
        returns = compute_gae(next_value, rewards, masks, values)
        returns = torch.cat(returns).detach() # concatenates along existing dimension and detach the tensor from the network graph, making the tensor no gradient
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values # compute advantage for each action
        advantage = normalize(advantage) # compute the normalization of the vector to make uniform values
        ppo_update(states, actions, log_probs, returns, advantage)
        train_epoch += 1


        if train_epoch % T_EPOCHS == 0: # do a test every T_EPOCHS times

            test_reward = np.mean([test_env(i, env, model, device) for i in range(N_TESTS)]) # do N_TESTS tests and takes the mean reward
            test_rewards.append(test_reward) # collect the mean rewards for saving performance metric
            test_epochs.append(train_epoch)
            print('Epoch: %s -> Reward: %.3f' % (train_epoch, test_reward))

            if best_reward is None or best_reward < test_reward: # save a checkpoint every time it achieves a better reward
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" %(best_reward, test_reward))
                    name = "%s_%+.3f_%d.dat" % (ENV_ID, test_reward, train_epoch)
                    fname = os.path.join('.', 'ppo_test/checkpoints', name)
                    states = {
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'test_rewards': test_rewards,
                      'test_epochs': test_epochs,
                    }
                    torch.save(states, fname) # save the model, for transfer learning is important to save: model parameters, optimizer parameters, epochs and rewards record as well

                best_reward = test_reward

            if test_reward > TARGET_REWARD: # stop training if archive the best
                early_stop = True


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available() # Autodetect CUDA
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    envs = [make_env() for i in range(N)] # Prepare N actors in N environments
    envs = SubprocVecEnv(envs) # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_inputs = 1
    num_outputs = envs.action_space.n

    model = CNN(num_inputs, num_outputs, H_SIZE).to(device)

    optimizer = optim.Adam(model.parameters(), lr=L_RATE) # implements Adam algorithm
    test_rewards = []
    test_epochs = []
    train_epoch = 0
    best_reward = None



    if TRANSFER_LEARNING: # transfer learning set this variable to True
      if use_cuda:
        checkpoint = torch.load(BASELINE_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        test_rewards=checkpoint['test_rewards']
        test_epochs=checkpoint['test_epochs']
        train_epoch=test_epochs[-1]
        best_reward=test_rewards[-1]

      else:
        checkpoint = torch.load(BASELINE_PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        test_rewards=checkpoint['test_rewards']
        test_epochs=checkpoint['test_epochs']
        train_epoch=test_epochs[-1]
        best_reward=test_rewards[-1]
      print('Baseline: loaded')
      print('Previous best reward: %.3f'%(best_reward))

    print(model)
    print(optimizer)

    losing_games_file = "losing_game.out"
    winning_games_file = "winning_game.out"
    ratio = 0.5

    ppo_train(model, envs, device, use_cuda, test_rewards, test_epochs, train_epoch, best_reward, losing_games_file, winning_games_file, ratio)



