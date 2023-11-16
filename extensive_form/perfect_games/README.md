# Environment

The three perfect games use same library OpenSpiel and have similar code structures. Please install OpenSpiel before executing any program. 

# Code Structure 

We provide the code to train a target agent in the `baseline/` folder of each game.

- `alpha_zero.py`: main script to execute training a mask network or testing the mask network.
- `ppo_gmax.py`: implement the learning algorithm.
- `mask_net.py`: implement the network of masknet.
- `alpha_zero_test.py`: test the target agent's performance and the masknet's performance.
- `alpha_zero_visualize.py`: execute visualizing a trajectory.
- `visualize.py`: visualize a trajectory.
- `attack.py`: launch adversarial attacks based on explanation.

# StateMask

- Run `alpha_zero.py` to train the masknet and do a fidelity test. Set `FLAGS.is_training=True` for training the masknet. Set `FLAGS.is_training=False` for the fidelity test.
 - We provide the code for visualization in `alpha_zero_visualize.py`.
