from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import app, flags, logging
import numpy as np



FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_float("threshold", 0.1, "threshold for selecting critical steps")
flags.DEFINE_integer("num_episodes", 500, "number of episodes")
flags.FLAGS(sys.argv)



def select_steps(path, critical, import_thrd):
  if critical:
    critical_steps_starts = []
    critical_steps_ends = []
  else:
    non_critical_steps_starts = []
    non_critical_steps_ends = []

  for i_episode in range(FLAGS.num_episodes):
    values_path = path + "mask_probs_" + str(i_episode) + ".out" 
    values = np.loadtxt(values_path)

    confs = values

    iteration_ends_path =  path + "eps_len_" + str(i_episode) + ".out"
    iteration_ends = np.loadtxt(iteration_ends_path)

    sorted_idx = np.argsort(confs)

    k = max(int(len(values) * import_thrd),1)
    idx = sorted_idx[-k:] if critical else sorted_idx[:k]
    idx.sort()

    steps_start, steps_end = idx[0], idx[0]
    ans, count = 0, 0
    tmp_end, tmp_start = idx[0], idx[0]

    for i in range(1, len(idx)):
      if idx[i] == idx[i - 1] + 1:
        count += 1
        tmp_end = idx[i]
      else:
        count = 0
        tmp_start = idx[i]
        tmp_end = idx[i]
      if count > ans:
        ans = count
        steps_start = tmp_start
        steps_end = tmp_end

    if critical:
      critical_steps_starts.append(steps_start)
      critical_steps_ends.append(steps_end)

    else:
      non_critical_steps_starts.append(steps_start)
      non_critical_steps_ends.append(steps_end)
      
  if critical:
    np.savetxt(path + str(int(FLAGS.threshold * 10)) + "critical_steps_starts.out", critical_steps_starts)
    np.savetxt(path + str(int(FLAGS.threshold * 10)) + "critical_steps_ends.out", critical_steps_ends)
  else:
    np.savetxt(path + str(int(FLAGS.threshold * 10)) + "non_critical_steps_starts.out", non_critical_steps_starts)
    np.savetxt(path + str(int(FLAGS.threshold * 10)) + "non_critical_steps_ends.out", non_critical_steps_ends)

def main(argv):
    path = '/data/jiahao/TencentSC2/StarCraftII/src/results/'
    select_steps(path, critical=True, import_thrd=FLAGS.threshold)
    select_steps(path, critical=False, import_thrd=FLAGS.threshold)



if __name__ == '__main__':
    app.run(main)
