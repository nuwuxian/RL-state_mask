from ast import Num
import numpy as np



def evaluate_fidelity_score(name,num):
    if name=='replay':
        critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
        critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")

        replay_rewards = np.loadtxt("./recording/replay_reward_record.out")
        original_rewards = np.loadtxt("./recording/reward_record.out")
    elif name=='replay_non':
        critical_steps_starts = np.loadtxt("./recording/non_critical_steps_starts.out")
        critical_steps_ends = np.loadtxt("./recording/non_critical_steps_ends.out")

        replay_rewards = np.loadtxt("./recording/replay_non_reward_record.out")
        original_rewards = np.loadtxt("./recording/reward_record.out")
    elif name=='rand_replay':
        critical_steps_starts = np.loadtxt("./recording/critical_steps_starts.out")
        critical_steps_ends = np.loadtxt("./recording/critical_steps_ends.out")

        replay_rewards = np.loadtxt("./recording/replay_rand_reward_record.out")
        original_rewards = np.loadtxt("./recording/reward_record.out")
    elif name=='rand_replay_non':
        critical_steps_starts = np.loadtxt("./recording/non_critical_steps_starts.out")
        critical_steps_ends = np.loadtxt("./recording/non_critical_steps_ends.out")

        replay_rewards = np.loadtxt("./recording/replay_rand_non_reward_record.out")
        original_rewards = np.loadtxt("./recording/reward_record.out")


    p_ls = []
    p_ds = []
    diff = 0
    fidelity_scores = []

    for i_episode in range(num):
        # if original_rewards[i_episode] == 0:
        #     continue
        iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
        iteration_ends = np.loadtxt(iteration_ends_path)

        critical_frames_start = critical_steps_starts[i_episode]
        critical_frames_end = critical_steps_ends[i_episode]
        random_replacement_steps = critical_frames_end - critical_frames_start + 1
        p_ls.append(random_replacement_steps/iteration_ends)
        
        p_ds.append(abs(replay_rewards[i_episode] - original_rewards[i_episode])/2)
        fidelity_scores.append(np.log(p_ls[-1]) - np.log(p_ds[-1] + 1e-3))
        if abs(replay_rewards[i_episode] - original_rewards[i_episode])/2 != 0:
            diff+=1
    p_l = np.mean(p_ls)
    p_d = np.mean(p_ds)

    print(p_l)
    print(np.log(p_l) - np.log(p_d))
    print(np.mean(fidelity_scores))
    print(diff)
    print("~~~~~~~~~~~~~~~")


# #############################################################################
if __name__ == "__main__":
    num=500
    evaluate_fidelity_score('replay', num)

