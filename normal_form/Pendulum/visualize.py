import gym
import numpy as np
from ppo_torch import Agent
from ppo_lasso import Masknet
from utils import plot_learning_curve
import os
import torch as T
from stable_baselines3 import PPO
from gym.wrappers.monitoring import video_recorder


def vis(agent, masknet, env, n_games=5):


    score_history = []

    n_steps = 0

    for i in range(n_games):
        vid_path = "./vis/vid_base_" + str (i) + ".mp4"
        vid = video_recorder.VideoRecorder(env,path=vid_path)
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

            vid.capture_frame()
            agent_action, _states = agent.predict(observation)
            mask_dist, mask_val = masknet.choose_action(observation)
            mask_action = mask_dist.sample()
            mask_prob = T.squeeze(mask_dist.log_prob(mask_action)).item().exp()
            mask_action = T.squeeze(mask_action).item()
            

            mask_probs.append(mask_prob)

            #if mask_action == 1:
            #    action = agent_action
            #else:
            #    num_mask += 1
            #    mask_pos.append(traj_len)
            #    action = env.action_space.sample()
            
            action = agent_action
            action_seq.append(action)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
   

            observation = observation_
            traj_len += 1
        
        vid.capture_frame()
        vid.close()
        print("traj " + str(i) + ": " + str(traj_len))
        print("num of mask: " + str(num_mask))
        score_history.append(score)


        print('score %.4f' % score)

        eps_len_filename = "./vis/eps_len_" + str(i) + ".out" 
        np.savetxt(eps_len_filename, [traj_len])

        act_seq_filename = "./vis/act_seq_" + str(i) + ".out" 
        np.savetxt(act_seq_filename, action_seq)

        mask_probs_filename = "./vis/mask_probs_" + str(i) + ".out" 
        np.savetxt(mask_probs_filename, mask_probs)

    print("=====Test mask network=====")
    print("Average score: ", np.mean(score_history))
    np.savetxt("./vis/reward_record.out", score_history)





if __name__ == '__main__':

    if os.path.isdir("vis"):
        os.system("rm -rf vis")

    os.system("mkdir vis")
    env = gym.make('Pendulum-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = PPO.load("./baseline/Pendulum-v0")

    masknet = Masknet(n_actions=2, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    masknet.load_models()

    vis(agent, masknet, env)


