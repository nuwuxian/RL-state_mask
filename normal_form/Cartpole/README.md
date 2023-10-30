*Environment*

Please install the gym environment before executing any program. We use the `CartPole-v1` environment for our experiment.

*Target Agent*

We provide a pre-trained target agent in `./baseline/CartPole-v1.zip` folder.

*StateMask*

- Run `main.py` to train the masknet. We give one trained masknet model in `./tmp/ppo/`. It can be used to calculate the fidelity test score or loaded as a pre-trained masknet model.
- Run `test.py` to test the performance of the target agent and the masknet.
- To do fidelity tests, execute the following programs:
- - `python test.py` to generate trajectories for explanation
  - `python select_critical_steps.py ` to select most important time steps
  - `python replay.py` to randomize actions in the identified critical steps
  - `python fidelity_score.py` to calculate the fidelity score.
- We provide the code for applications such as attack in `attack.py`.
