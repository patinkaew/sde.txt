"""
Sampling from DDPM with stochastic inverse process.
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
    config_path = 'celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    save_path = 'result/celeba-0'
    log_every = 10
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # sampler = diff.sample_stochastic_step
    sampler = diff.sample_deterministic_step

    print('Set up...')

    # Set up
    util.mkdir_if_not_exists(save_path)

    # Set up parameters
    config = util.load_config(config_path)
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, device)
    std = torch.exp(0.5 * logvar)
    
    # Load model
    model = Model(config)
    util.load_model(model, ckpt_path, device)

    print('Start sampling...')

    # Sampling
    x = torch.randn(1, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in reversed(range(num_time_steps)):
        if not (t + 1) % log_every:
            print('Time step {}'.format(t))
            torch.save(x, os.path.join(save_path, '{}.pt'.format(t)))
            util.save_image(x, os.path.join(save_path, '{}.png'.format(t)))
        x = sampler(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], betas[t], std[t])

if __name__ == '__main__':
    main()