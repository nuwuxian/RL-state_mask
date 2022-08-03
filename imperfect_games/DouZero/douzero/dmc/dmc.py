import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .masknet import MaskNet
from .utils import get_buffer, log, create_env, create_buffers, create_optimizers, act

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

GAMMA = 0.99 
LAM = 0.95
clip_param = 0.2

C_1 = 0.5 # squared loss coefficient
C_2 = 0.01 # entropy coefficient


def merge(buffer_list):
    ret_buffer = {
        key: torch.stack([buffer_list[i][key] for m in indices], dim=1)
        for key in buffer_list[0]
    }
    return ret_buffer

def sample(buffer, sample_sz, max_sz):
    ret_sample = {}
    sample_id = random.sample(np.arange(max_sz), sample_sz)
    for k in batch:
        ret_sample[k] = buffer[k][sample_id, ...]
    return ret_sample

 def checkpoint(frames):
    if flags.disable_checkpoint:
        return
    log.info('Saving checkpoint to %s', checkpointpath)
    _model = learner_model.get_model()
    torch.save({
        'model_state_dict': _model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "stats": stats,
        'flags': vars(flags),
        'frames': frames,
        'position_frames': position_frames
    }, checkpointpath)

    # Save the weights for evaluation purpose
    model_weights_dir = os.path.expandvars(os.path.expanduser(
        '%s/%s/%s' % (flags.savedir, flags.xpid, flags.position+'_masknet_weights_'+str(frames)+'.ckpt')))
    torch.save(learner_model.get_model().state_dict(), model_weights_dir)


def learn(model, batch, optimizer, flags):
    """PPO update."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    position = flags.position
    obs_z = batch['obs_z'].to(device)
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    act = batch['act'].to(device)
    log_probs = batch['logpac'].to(device)
    ret = batch['ret'].to(device)
    adv = batch['adv'].to(device)

    episode_returns = batch['reward'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    
    dist, value = model(obs_z, obs_x_no_action)
    new_log_probs = dist.log_prob(act)
    ratio = (new_log_probs - log_probs).exp() # new_prob/old_prob
    surr1 = ratio * advs
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
    actor_loss = - torch.min(surr1, surr2).mean()
    critic_loss = (ret - value).pow(2).mean()
    entropy = dist.entropy().mean()
    loss = C_1 * critic_loss + actor_loss - C_2 * entropy
    stats = {
        'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
        'loss_'+position: loss.item(),
    }
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
    optimizer.step()
    return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    position = flags.position
    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models, mask_models = {}, {}
    for device in device_iterator:
        # create baseline model
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model
        # create masknet model
        masknet = MaskNet(device=device, postion=position)
        masknet.share_memory()
        masknet.eval()
        mask_models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        _free_queue, _full_queue = ctx.SimpleQueue(), ctx.SimpleQueue()
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Masknet(device=flags.training_device, position=position)

    # Create optimizer
    optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], mask_models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)
    # Only one learner process
    def batch_and_learn():
        fps_log = []
        timer = timeit.default_timer
        last_checkpoint_time = timer() - flags.save_interval * 60
        nonlocal frames, stats
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            # Merge all the buffers
            device_buffer = []
            for device in device_iterator:
                buffer = get_buffer(free_queue[device], full_queue[device], buffers[device], flags)
                device_buffers.append(buffer)
            x_buffer = merge(device_buffer)
            # ppo update
            for i in range(flags.num_epochs):
                for _ in range(flags.nminibatches):
                    sample_sz = int(T * B * flags.num_actor_devices / flags.nminibatches)
                    batch = sample(x_buffer, sample_sz, T * B * * flags.num_actor_devices)
                    _stats = learn(learner_model.get_model(), batch, optimizer, flags)
                    for k in _stats:
                        stats[k] = _stats[k]
                
            to_log = dict(frames=frames)
            to_log.update({k: stats[k] for k in stat_keys})
            plogger.log(to_log)    
            frames += T * B * flags.num_actor_devices
            # Broadcast the newly update masknet
            for mask_model in mask_models.values():
                mask_model.get_model().load_state_dict(model.state_dict())

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)
            
            log.info('After %i frames: @ %.1f fps (avg@ %.1f fps) Stats:\n%s',
                     frames,
                     fps,
                     fps_avg,
                     pprint.pformat(stats))

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)
    # Train masknet
    batch_and_learn()
