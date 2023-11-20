# Environment
Please install the MuJoCo environment before executing any program. We use the `multicomp/YouShallNotPassHumans-v0` environment for our experiment.

# Code Structure
- `src/mask_train.py`: main script to execture training a mask network.
- `src/victim_retrain.py`: main script to retrain the target agent (i.e., the runner).
- `src/attack.py`: launch attacks based on explanation.
- `src/traj.py`: test the performance of the masknet.
- `src/replay.py`: replay the target agent based on explanation.
- `src/fidelity_score.py`: calculate the masknet's fidelity.

# StateMask
- run `src/mask_train.py` or `src/run.sh` to train the masknet.
- run `src/victim_train.py` to retrain the agent.
- run `src/traj.py`, `src/select_critical_steps.py` , `src/run_play.sh`, and `src/fidelity_score.py` to calculate the fidelity of the masknet.