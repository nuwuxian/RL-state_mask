# ExpMask: Explainable Deep Reinforcement Learning through Action Mask

This repo contains the code for the paper titled "ExpMask: Explainable Deep Reinforcement Learning through Action Mask".

## Requirement
Most of the codebase is written for ```python3.7 and Pytorch``` except for `You-Shall-Not-Pass` and `Kick-And-Defend` which requires `TensorFlow`. If you run errors in some programs, install the missing lib via pip install as the error report. 

## Code Structure and instructions
### Basics
- We implement our methods in ten reinforcement learning game environments. Six of them are normal-form games while the other four are extensive-form games.
- Normal-form games contain the following six games: Pong, You-Shall-Not-Pass, Kick-And-Defend, CartPole, Pendulum, and StarCraft II.
- Extensive-form games contain the following four games: Connect 4, Tic-Tac-Toe, Breakthrough, and DouDizhu. The first three belongs to perfect-information games while the last is a imperfect-information game.

### Training
- To run all the code, please refer to the readme in the corresponding subfolder.
