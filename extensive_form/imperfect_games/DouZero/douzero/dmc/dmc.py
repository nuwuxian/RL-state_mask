import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import random
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from .file_writer import FileWriter
from .models import Model
from .masknet import MaskNet
from .utils import get_buffer, log, create_env, create_buffers, act

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}
clip_param = 0.2

C_1 = 0.5 # squared loss coefficient
C_2 = 0.0 # entropy coefficient

def merge(buffer_list):
    sz = len(buffer_list)
    ret_buffer = {
        key: torch.cat([buffer_list[i][key] for i in range(sz)])
        for key in buffer_list[0]
    }
    return ret_buffer

def sample(buffer, sample_sz, max_sz):
    ret_sample = {}
    sample_id = random.sample(range(max_sz), sample_sz)
    for k in buffer:
        ret_sample[k] = buffer[k][sample_id, ...]
    return ret_sample

def learn(model, batch, optimizer, flags):
    """PPO update."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    position = flags.position
    obs_z = batch['obs_z'].to(device).float()
    obs_x_no_action = batch['obs_x_no_action'].to(device).float()
    act = batch['act'].to(device)
    log_probs = batch['logpac'].to(device)
    ret = batch['ret'].to(device)
    adv = batch['adv'].to(device)

    # normalize the adv
    adv = (adv - torch.mean(adv,dim=0))/(1e-7 + torch.std(adv,dim=0))

    episode_returns = batch['reward'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    
    dist, value = model.forward(obs_z, obs_x_no_action)
    new_log_probs = dist.log_prob(act)
    ratio = (new_log_probs - log_probs).exp() # new_prob/old_prob
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
    actor_loss = - torch.min(surr1, surr2).mean()
    critic_loss = (ret - value).pow(2).mean()
    entropy = dist.entropy().mean()
    loss = C_1 * critic_loss + actor_loss - C_2 * entropy

    avg_return = torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item()
    avg_mask_ratio = act.float().mean().item()
    # update optimizer 
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
    optimizer.step()

    return loss.item(), critic_loss.item(), actor_loss.item(), entropy.item(), avg_mask_ratio, avg_return

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
    
    savedir = flags.savedir + '/' + 'LR_' + str(flags.learning_rate) + '_NUM_EPOCH_' + str(flags.num_epochs) + '_NMINIBATCHES_' + str(flags.nminibatches)

    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (savedir, flags.xpid, 'model.tar')))
    writer = SummaryWriter(log_dir=savedir)

    position = flags.position
    pretrain_path = flags.pretrain_path
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
        masknet = MaskNet(device=device, position=position)
        masknet.share_memory()
        masknet.eval()
        mask_models[device] = masknet

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
    learner_model = MaskNet(device=flags.training_device, position=position)

    # Create optimizer
    optimizer = torch.optim.Adam(
            learner_model.parameters(),
            lr=flags.learning_rate)
    frames = 0
    # Load models if any
    if flags.load_model and os.path.exists(pretrain_path):
        checkpoint_states = {}
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            checkpoint_states[k] = torch.load(
                pretrain_path + '/' + k + '.ckpt', map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
            )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(checkpoint_states[k])
        log.info(f"Load baseline pretrained models")

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _model = learner_model.get_model()
        torch.save({
            'model_state_dict': _model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'flags': vars(flags),
            'frames': frames,
        }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (savedir, flags.xpid, flags.position+'_masknet_weights_'+str(frames)+'.ckpt')))
        torch.save(learner_model.get_model().state_dict(), model_weights_dir)
    
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
        nonlocal frames
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            # Merge all the buffers
            device_buffer = []
            for device in device_iterator:
                buffer = get_buffer(free_queue[device], full_queue[device], buffers[device], flags)
                device_buffer.append(buffer)
            x_buffer = merge(device_buffer)
            # ppo update
            avg_loss, avg_actor_loss, avg_critic_loss, avg_entropy_loss = [], [], [], []
            avg_mask_ratio, avg_return = [], []

            for i in range(flags.num_epochs):
                for _ in range(flags.nminibatches):
                    sample_sz = int(T * B * flags.num_actor_devices / flags.nminibatches)
                    batch = sample(x_buffer, sample_sz, T * B * flags.num_actor_devices)
                    _loss, _critic_loss, _actor_loss, _entropy_loss, _avg_mask_ratio, _avg_return = learn(learner_model, batch, optimizer, flags)
                    avg_loss.append(_loss)
                    avg_critic_loss.append(_critic_loss)
                    avg_actor_loss.append(_actor_loss)
                    avg_entropy_loss.append(_entropy_loss)
                    avg_mask_ratio.append(_avg_mask_ratio)
                    avg_return.append(_avg_return)

            frames += T * B * flags.num_actor_devices
            # Broadcast the newly update masknet
            for mask_model in mask_models.values():
                mask_model.get_model().load_state_dict(learner_model.get_model().state_dict())
            # Clear the free_queue and full_queue
            for device in device_iterator:
                while not full_queue[device].empty(): full_queue[device].get()
                while not free_queue[device].empty(): free_queue[device].get()
                for m in range(flags.num_buffers):
                    free_queue[device].put(m)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

             # writer into logger
            writer.add_scalar('Loss', np.mean(avg_loss), global_step=frames)
            writer.add_scalar('Loss_critic', np.mean(avg_critic_loss), global_step=frames)
            writer.add_scalar('Loss_actor', np.mean(avg_actor_loss), global_step=frames)
            writer.add_scalar('Loss_entropy', np.mean(avg_entropy_loss), global_step=frames)
            writer.add_scalar('Mask_ratio', 1 - np.mean(avg_mask_ratio), global_step=frames)
            writer.add_scalar('Return', np.mean(avg_return), global_step=frames)
            writer.add_scalar('FPS_avg', fps_avg, global_step=frames)
            # logger into the monitor
            log.info('Training %i frames: FPS_avg: %.3f Loss: %.3f Loss_critic: %.3f Loss_actor: %.3f Loss_entropy: %.3f \
                Mask_ratio: %.3f Return: %.3f', frames, fps_avg, np.mean(avg_loss), np.mean(avg_critic_loss), np.mean(avg_actor_loss), \
                np.mean(avg_entropy_loss), 1 - np.mean(avg_mask_ratio), np.mean(avg_return))

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)
    # Train masknet
    batch_and_learn()
