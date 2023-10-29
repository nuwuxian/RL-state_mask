
*Target Agent*

We provide a pre-trained target agent in `./ppo_test/baseline/Pong-v0_+0.896_12150.dat` folder.

*StateMask*

- Run `ppo.py` to train the masknet. To train from scratch, set `TRANSFER_LEARNING=False` in `ppo.py`.
- We give one trained masknet model in `./ppo_test/checkpoints/Pong-v0_+0.855_19700.dat`. It can be used to calculate the fidelity test score or loaded as a pre-trained masknet model (set `TRANSFER_LEARNING=True` in `ppo.py`).
