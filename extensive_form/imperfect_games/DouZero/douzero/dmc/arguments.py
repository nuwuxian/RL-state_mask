import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='wp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=1, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', default=True, type=bool,
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')
parser.add_argument('--pretrain_path', default='./baselines/douzero_WP')

# Hyperparameters
parser.add_argument('--total_frames', default=30000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=42, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--max_grad_norm', default=0.5, type=float,
                    help='Max norm of gradients')
parser.add_argument('--position', default='landlord', type=str,
                    help='explain position')
parser.add_argument('--reward_bonus_coeff', default=0, type=float,
                    help='The coefficient of bonus in the reward')
parser.add_argument('--lasso_coeff', default=0.06, type=float,
                    help='The coefficient of lasso in the loss')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0003, type=float,
                    help='Learning rate')

parser.add_argument('--anneal_rl', default=True, type=bool)
parser.add_argument('--fix_lr', default=False, type=bool)
parser.add_argument('--step_lr', default=False, type=bool)
parser.add_argument('--num_epochs', default=4, type=int,
                    help='PPO inner training epochs')
parser.add_argument('--nminibatches', default=4, type=int,
                    help='PPO inner number of mini-batches')
