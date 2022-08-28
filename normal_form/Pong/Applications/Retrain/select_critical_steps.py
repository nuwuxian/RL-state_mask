import numpy as np

critical_steps_starts = []
critical_steps_ends = []

for i_episode in range(500):
    mask_probs_path = "./recording/mask_probs_" + str(i_episode) + ".out"
    mask_probs = np.loadtxt(mask_probs_path)

    confs = mask_probs[:,1]

    iteration_ends_path = "./recording/eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    k = int(iteration_ends * 0.3)

    #find the top k:
    idx = np.argpartition(confs, -k)[-k:]  # Indices not sorted


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

np.savetxt("./recording/critical_steps_starts.out", critical_steps_starts)
np.savetxt("./recording/critical_steps_ends.out", critical_steps_ends)
