import gym
import numpy as np
from ppo_lasso import Masknet
from utils import plot_learning_curve
import torch as T
from stable_baselines3 import PPO

#eta_origin = 0.2

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = PPO.load("./baseline/CartPole-v1")


    masknet = Masknet(n_actions=2, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 1000

    figure_file = 'plots/masknet.png'
    figure_file_2 = 'plots/eta.png'

    best_score = env.reward_range[0]
    score_history = []
    discounted_reward_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        discounted_reward = 0

        num_mask = 0
        traj_len = 0
        count = 0


        while not done:
            agent_action, _states = agent.predict(observation)

            mask_dist, mask_val = masknet.choose_action(observation)
            mask_action = mask_dist.sample()
            
            mask_prob = T.squeeze(mask_dist.log_prob(mask_action)).item()
            mask_val = T.squeeze(mask_val).item()
            
            mask_action = T.squeeze(mask_action).item()


            if mask_action == 1:
                action = agent_action
            else:
                num_mask += 1
                #action = np.random.choice(env.action_space.n)
                action = env.action_space.sample()
            
            observation_, reward, done, info = env.step(action)
            discounted_reward += np.power(0.99, count) * reward
            n_steps += 1
            count += 1
            score += reward
            masknet.remember(observation, mask_action, mask_prob, mask_val, reward, done)               
            observation = observation_
            traj_len += 1

        masknet.learn(num_mask, discounted_reward)  
        learn_iters += 1
        print("traj " + str(i) + ": " + str(traj_len))
        print("num of mask: " + str(num_mask))
        score_history.append(score)
        discounted_reward_history.append(discounted_reward)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            masknet.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    np.savetxt("final_reward.out", score_history)
    np.savetxt("discounted_reward.out", discounted_reward_history)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    plot_learning_curve(x, discounted_reward_history, figure_file_2)
