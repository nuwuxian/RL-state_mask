import numpy as np

critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")

replay_rewards = np.loadtxt("./recording/replay_reward_record.out")
original_rewards = np.loadtxt("./recording/reward_record.out")
print(np.mean(original_rewards))

p_ls = []
p_ds = []
fidelity_scores = []

for i_episode in range(500):
    iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    critical_frames_start = critical_steps_starts[i_episode]
    critical_frames_end = critical_steps_ends[i_episode]
    random_replacement_steps = critical_frames_end - critical_frames_start

    p_ls.append(random_replacement_steps/iteration_ends)
    p_ds.append(abs(replay_rewards[i_episode] - original_rewards[i_episode]))

p_l = np.mean(p_ls)
p_d = np.mean(p_ds/np.max(p_ds))

print(np.log(p_l) - np.log(p_d))
