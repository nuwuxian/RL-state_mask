import numpy as np

non_critical_steps_starts = []
non_critical_steps_ends = []

for i_episode in range(500):
    mask_probs_path = "./recording/mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs[:,1]

    iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    k = int(iteration_ends * 0.1)

    #find the top k:
    #idx = np.argpartition(confs, -k)[-k:]  # Indices not sorted

    #sorted_idxs = idx[np.argsort(confs[idx])][::-1] # Indices sorted by value from largest to smallest
    #print(sorted_idxs)
    #print(confs[sorted_idxs])

    #find the bottom k:

    idx = np.argpartition(confs, k)[:k]  # Indices not sorted

    #idx[np.argsort(x[idx])]  # Indices sorted by value from smallest to largest

    idx.sort()

    non_critical_steps_start = idx[0]
    non_critical_steps_end = idx[0]

    ans = 0
    count = 0

    tmp_end = idx[0]
    tmp_start = idx[0]

    for i in range(1, len(idx)):
     
        if idx[i] == idx[i - 1] + 1:
            count += 1
            tmp_end = idx[i]
             
        # Reset the count
        else:
            count = 0
            tmp_start = idx[i]
            tmp_end = idx[i]
             
        # Update the maximum
        if count > ans:
            ans = count
            non_critical_steps_start = tmp_start
            non_critical_steps_end = tmp_end

         

    non_critical_steps_starts.append(non_critical_steps_start)
    non_critical_steps_ends.append(non_critical_steps_end)

np.savetxt("./recording/non_critical_steps_starts.out", non_critical_steps_starts)
np.savetxt("./recording/non_critical_steps_ends.out", non_critical_steps_ends)
