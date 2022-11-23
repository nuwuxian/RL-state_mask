import numpy as np

critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")

replay_rewards = np.loadtxt("./recording/replay_reward_record.out")
original_rewards = np.loadtxt("./recording/reward_record.out")


fidelity_scores = []

for i_episode in range(500):
    
    iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    critical_frames_start = critical_steps_starts[i_episode]
    critical_frames_end = critical_steps_ends[i_episode]
    random_replacement_steps = critical_frames_end - critical_frames_start + 1

    p_l = random_replacement_steps/iteration_ends

    tmp = abs(replay_rewards[i_episode] - original_rewards[i_episode])/1
    p_d = tmp if tmp>0 else 0.001

    fidelity_scores.append(np.log(p_l) - np.log(p_d))


print(np.mean(fidelity_scores))