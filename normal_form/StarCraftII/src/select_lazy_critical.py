import numpy as np

critical_steps_starts = []
critical_steps_ends = []

def lazy_gap(traj, iteration_ends):
    q_values = []
    gap_traj = []
    for i in range(10):
        lazy_gap_path = "./lazy_recording/" + str(i) + "/" + str(traj) + ".out"
        q_values.append(np.loadtxt(lazy_gap_path))
    for i in range(int(iteration_ends)-1):
        tmp = []
        for j in range(10):
            tmp.append(q_values[j][i])
        tmp = np.array(tmp)
        gap = np.max(tmp) - np.mean(tmp)
        gap_traj.append(gap)
    gap_traj = np.array(gap_traj)
    return gap_traj
            

for i_episode in range(50):
    iteration_ends_path = "./results/eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    lazy_gap_value = lazy_gap(i_episode, iteration_ends)
    confs = lazy_gap_value

    k = int(iteration_ends * 0.1)

    #find the top k:
    idx = np.argpartition(confs, -k)[-k:]  # Indices not sorted

    sorted_idxs = idx[np.argsort(confs[idx])][::-1] # Indices sorted by value from largest to smallest
    #print(sorted_idxs)
    #print(confs[sorted_idxs])

    #find the bottom k:

    #idx = np.argpartition(x, k)[:k]  # Indices not sorted

    #idx[np.argsort(x[idx])]  # Indices sorted by value from smallest to largest

    idx.sort()

    critical_steps_start = idx[0]
    critical_steps_end = idx[0]

    ans = 0
    count = 0

    tmp_end = idx[0]
    tmp_start = idx[0]

    for i in range(1, len(idx)):
     
        # Check if the current element is
        # equal to previous element +1
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
            critical_steps_start = tmp_start
            critical_steps_end = tmp_end

         

    critical_steps_starts.append(critical_steps_start)
    critical_steps_ends.append(critical_steps_end)

np.savetxt("./results/1critical_steps_starts.out", critical_steps_starts)
np.savetxt("./results/1critical_steps_ends.out", critical_steps_ends)
