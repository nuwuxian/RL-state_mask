# Environment

Please run `pip install -r requirements.txt` before executing any program. The code is based on DouZero repository (https://github.com/kwai/DouZero)

# Code Structure 

We provide pre-trained target agents in `./baselines/douzero_WP/` folder. There are three agents in DouDizhu game and our target agent is the landlord.

- `train.py`: main script to execute training a mask network.
- `generate_eval_data.py': randomly allocate cards for each role.
- `fidelity_test.py`: test the masknet's performance.
- `visualize.py`: visualize a trajectory.
- `attack.py`: launch adversarial attacks based on explanation.

# StateMask

- Run `train.py` to train the masknet. We give one trained masknet model in `./masknet/landlord_masknet_weights_19475400.ckpt`. It can be used to calculate the fidelity test score or loaded as a pre-trained masknet model.
- Run `fidelity_test.py` to do fidelity tests.
- We provide the code for applications such as attack in `attack.py`.
