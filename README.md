# StateMask: Explainable Deep Reinforcement Learning through State Mask

This repo contains the code for the paper "StateMask: Explainable Deep Reinforcement Learning through State Mask" (https://openreview.net/forum?id=pzc6LnUxYN).

Paper citation:
```
@inproceedings{cheng2023statemask,
title={StateMask: Explainable Deep Reinforcement Learning through State Mask},
author={Zelei Cheng and Xian Wu and Jiahao Yu and Wenhai Sun and Wenbo Guo and Xinyu Xing},
booktitle={Proc. of NeurIPS},
year={2023}
```

## Requirement
Most of the codebase is written for ```python3.7``` and ```Pytorch``` except for `You-Shall-Not-Pass` and `Kick-And-Defend` which requires `TensorFlow`. Most of the games require `gym==0.19.0`. Installing the requirements of `You-Shall-Not-Pass` and `Kick-And-Defend` could refer to https://github.com/openai/multiagent-competition. If you run errors in some programs, install the missing lib via pip install as the error report. 

## Introduction
### Basics
- We implement our methods in ten reinforcement learning game environments. Six of them are normal-form games while the other four are extensive-form games.
- Normal-form games contain the following six games: Pong, You-Shall-Not-Pass, Kick-And-Defend, CartPole, Pendulum, and StarCraft II.
- Extensive-form games contain the following four games: Connect 4, Tic-Tac-Toe, Breakthrough, and DouDizhu. The first three belongs to perfect-information games while the last is a imperfect-information game.
- In each game, we provide code for training and evaluating the explanation mask and retraining the target agent under the guidance of explanation.

### Usage
- To run all the experiments, please refer to the README for detailed instructions in each corresponding subfolder.
