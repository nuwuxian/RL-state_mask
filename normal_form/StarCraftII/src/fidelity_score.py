from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from absl import app, flags, logging
import numpy as np



FLAGS = flags.FLAGS
# total time steps.
flags.DEFINE_float("threshold", 0.4, "threshold for selecting critical steps")
flags.DEFINE_integer("num_episodes", 20, "number of episodes")
flags.FLAGS(sys.argv)



def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios
    p_ds = []

    for j in range(len(p_ls)):
        p_ds.append(np.abs(results[j]-replay_results[j]))
    reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001

    fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
    return fid_score

def main(argv):
    path = '/data/jiahao/TencentSC2/StarCraftII/src/results/'
    original_reward = np.loadtxt(path + 'reward.out')
    replay_reward = np.loadtxt(path + '4critical_reward.out')
    ratio = []
    critical_steps_starts = np.loadtxt(path + "4critical_steps_starts.out")
    critical_steps_ends = np.loadtxt(path + "4critical_steps_ends.out")
    for i in range(FLAGS.num_episodes):
        original_len = np.loadtxt(path + 'eps_len_' + str(i) + '.out')
        ratio.append((critical_steps_ends[i] - critical_steps_starts[i] + 1) / original_len)
    ratio = np.array(ratio)
    score = cal_fidelity_score(ratio, original_reward, replay_reward)
    print(score)
    print(np.mean(ratio))


if __name__ == '__main__':
    app.run(main)
