*Environment*

The three perfect games use same library OpenSpiel and have similar code structures. Please install OpenSpiel before executing any program. 

*Target Agent*

We provide the code to train a target agent in the `baseline/` folder of each game.

*StateMask*

- Run `alpha_zero.py` to train the masknet and do a fidelity test. Set `FLAGS.is_training=True` for training the masknet. Set `FLAGS.is_training=False` for the fidelity test.
 - We provide the code for visualization in `alpha_zero_visualize.py`.
