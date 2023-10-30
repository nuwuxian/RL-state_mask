*Environment*

Please run `pip install -r requirements.txt` before executing any program. The code is based on DouZero repository (https://github.com/kwai/DouZero)

*Target Agent*

We provide pre-trained target agents in `./baselines/douzero_WP/` folder. There are three agents in DouDizhu game and our target agent is the landlord.

*StateMask*

- Run `train.py` to train the masknet. We give one trained masknet model in `./masknet/landlord_masknet_weights_19475400.ckpt`. It can be used to calculate the fidelity test score or loaded as a pre-trained masknet model.
- Run `fidelity_test.py` to do fidelity tests.
- We provide the code for applications such as attack in `attack.py`.
