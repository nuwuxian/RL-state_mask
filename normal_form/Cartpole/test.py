import gym
import numpy as np
from ppo_lasso import Masknet
from utils import plot_learning_curve
import os
import torch as T
from stable_baselines3 import PPO


def test_baseline(agent, env, n_games=500):

    score_history = []
    discounted_reward_history = []

    n_steps = 0

    for i in range(n_games):
        env.seed(i)
        observation = env.reset()
        done = False
        score = 0
        discounted_score = 0
        count = 0

        while not done:
            action, _states = agent.predict(observation)

            observation_, reward, done, info = env.step(action)
            discounted_score += np.power(0.99, count) * reward
            n_steps += 1
            count += 1
            score += reward           
            observation = observation_

        print('episode', i, 'score %.4f' % score) 
        score_history.append(score)
        discounted_reward_history.append(discounted_score)
        

    print("=====Test baseline network=====")
    print("Average score: ", np.mean(score_history))
    print("Expected total reward: ", np.mean(discounted_reward_history))



def test_mask(agent, masknet, env, n_games=500):


    score_history = []

    n_steps = 0

    for i in range(n_games):
        env.seed(i)
        observation = env.reset()
        done = False
        score = 0

        num_mask = 0
        traj_len = 0

        action_seq = []
        mask_pos = []
        mask_probs = []


        while not done:
            agent_action, _states = agent.predict(observation)


            mask_dist, mask_val = masknet.choose_action(observation)
            mask_action = mask_dist.sample()
            mask_prob = T.squeeze(mask_dist.log_prob(mask_action)).item()
            mask_action = T.squeeze(mask_action).item()
            

            mask_probs.append(mask_prob)

            if mask_action == 1:
                action = agent_action
            else:
                num_mask += 1
                mask_pos.append(traj_len)
                action = env.action_space.sample()
            
            action_seq.append(action)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
   

            observation = observation_
            traj_len += 1
        
        print("traj " + str(i) + ": " + str(traj_len))
        print("num of mask: " + str(num_mask))
        score_history.append(score)


        print('score %.4f' % score)
        
        mask_pos_filename = "./recording/mask_pos_" + str(i) + ".out" 
        np.savetxt(mask_pos_filename, mask_pos)

        eps_len_filename = "./recording/eps_len_" + str(i) + ".out" 
        np.savetxt(eps_len_filename, [traj_len])

        act_seq_filename = "./recording/act_seq_" + str(i) + ".out" 
        np.savetxt(act_seq_filename, action_seq)

        mask_probs_filename = "./recording/mask_probs_" + str(i) + ".out" 
        np.savetxt(mask_probs_filename, mask_probs)

    print("=====Test mask network=====")
    print("Average score: ", np.mean(score_history))
    np.savetxt("./recording/reward_record.out", score_history)





if __name__ == '__main__':

    if os.path.isdir("recording"):
        os.system("rm -rf recording")


    os.system("mkdir recording")
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = PPO.load("./baseline/CartPole-v1")



    masknet = Masknet(n_actions=2, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    masknet.load_models()


    test_baseline(agent, env)
    #test_mask(agent, masknet, env)


