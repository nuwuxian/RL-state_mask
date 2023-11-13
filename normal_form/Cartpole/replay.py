import gym
import numpy as np
from ppo_lasso import Masknet
from stable_baselines3 import PPO

def replay_important(agent, masknet, env, n_games=500):
    critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
    critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")


    replay_rewards = []

    n_steps = 0

    for i in range(n_games):
        env.seed(i)
        observation = env.reset()
        done = False
        score = 0

        num_mask = 0
        count = 0

        action_sequence_path = "./recording/act_seq_" + str(i) + ".out"
        recorded_actions = np.loadtxt(action_sequence_path)



        while not done:
            if count < critical_steps_starts[i]:
                observation_, reward, done, info = env.step(int(recorded_actions[count]))
            elif count <= critical_steps_ends[i]:
                observation_, reward, done, info = env.step(env.action_space.sample())
            else:
                agent_action, _states = agent.predict(observation)
                observation_, reward, done, info = env.step(agent_action)
            score += reward
            observation = observation_
            count += 1
        
        print("traj " + str(i) + ": " + str(count))
        replay_rewards.append(score)


        print('score %.4f' % score)
        

    print("=====Replay test (important)=====")
    print("Average score: ", np.mean(replay_rewards))
    np.savetxt("./recording/replay_reward_record.out", replay_rewards)

if __name__ == '__main__':
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


    replay_important(agent, masknet, env)

