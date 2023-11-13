import gym
import numpy as np
from ppo_torch import Agent
from ppo_lasso import Masknet
import os
import torch as T
from stable_baselines3 import PPO

threshold = 0.99

def test_baseline(agent, env, n_games=500):
    score_history = []
    n_steps = 0

    for i in range(n_games):

        env.seed(i)
        observation = env.reset()
        done = False
        score = 0


        while not done:
            action, _states = agent.predict(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            observation = observation_


        print('episode', i, 'score %.4f' % score) 
        score_history.append(score)

    return score_history



def attack(agent, masknet, env, n_games=500):
    score_history = []
    attack_ratios = []
    n_steps = 0

    for i in range(n_games):
        env.seed(i)
        observation = env.reset()
        done = False
        score = 0

        attack_num = 0
        traj_len = 0

        action_seq = []
        mask_probs = []


        while not done:
            agent_action, _states = agent.predict(observation)


            mask_dist, mask_val = masknet.choose_action(observation)
            mask_action = mask_dist.sample()
            mask_prob = T.squeeze(mask_dist.log_prob(T.Tensor([1]).cuda())).item()


            if np.exp(mask_prob) > threshold:
                action = env.action_space.sample()
                attack_num += 1
            else:
                action = agent_action
            
            action_seq.append(action)
            observation_, reward, done, info = env.step(action)
            score += reward
   

            observation = observation_
            traj_len += 1
        
        attack_ratio = attack_num/traj_len
        print("traj " + str(i) + ": ", traj_len)
        print('score %.4f' % score)
        print("attack ratio: ", attack_ratio)
        score_history.append(score)
        attack_ratios.append(attack_ratio)

    return score_history, attack_ratios



if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    eta_origin = 0.198
    agent = PPO.load("./baseline/CartPole-v1")

    masknet = Masknet(eta_origin=eta_origin, n_actions=2, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    masknet.load_models()

    score_history = test_baseline(agent, env)
    print("=====Before attack=====")
    print("Average score: ", np.mean(score_history))
    
    score_history, attack_ratios = attack(agent, masknet, env)
    
    print("=====After attack=====")
    print("Average score: ", np.mean(score_history))
    print("Average attack ratio: ", np.mean(attack_ratios))



