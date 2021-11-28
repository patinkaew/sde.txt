"""
Sampling from DDPM without text guidance.
"""

import os

import numpy as np
import torch

import diffusion as diff
from model import Model
import util

@torch.no_grad()
def main():

    # Arguments
    config_path = 'config_yml/celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    save_path = 'result/celeba-det-2'
    batch_size = 5
    log_every = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sampling_method = 'sto'

    print('Device: {}'.format(device))
    print('Set up...')

    # Set up
    util.mkdir_if_not_exists(save_path)

    # Set up parameters
    config = util.load_config(config_path)
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, device)
    std = torch.exp(0.5 * logvar) # only for stochastic sampling
    ones = torch.ones(batch_size, device=device)
    
    # Load model
    model = Model(config)
    util.load_model(model, ckpt_path, device)

    print('Start sampling...')

    # Sampling
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in reversed(range(num_time_steps)):
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t)
        if sampling_method == 'sto':
            x = diff.sample_stochastic_step(x, model, t, alphas_cumprod[t], betas[t], std[t], ones)
        elif sampling_method == 'det':
            x = diff.sample_deterministic_step(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        else:
            raise ValueError('Invalid sampling method')

if __name__ == '__main__':
    main()
