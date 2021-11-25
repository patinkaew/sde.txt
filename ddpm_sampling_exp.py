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
    save_path = 'result/celeba-0'
    log_every = 50

    # Set up
    util.mkdir_if_not_exists(save_path)

    # Set up parameters
    config = util.load_config(config_path)
    betas, alphas, alphas_cumprod, logvar, num_time_steps = diff.get_noise_schedule(config)
    std = torch.exp(0.5 * logvar)
    
    # Load model
    model = Model(config)

    # Sampling
    x = torch.randn(1, config.data.channels, 
                    config.data.image_size, config.data.image_size)
    for t in range(num_time_steps - 1, -1, -1):
        if not (t + 1) % log_every:
            print('Time step {}'.format(t))
            torch.save(x, os.path.join(save_path, '{}.pt'.format(t)))
            util.save_image(x, os.path.join(save_path, '{}.png'.format(t)))
        x = diff.sample_stochastic_step(x, model, t, alphas_cumprod[t], betas[t], std[t])

if __name__ == '__main__':
    main()