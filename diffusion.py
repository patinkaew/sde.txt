"""
Sampling with Diffusion model

Adapted from
    https://github.com/ermongroup/SDEdit/blob/c0ed910a759df68ecc373caa020f6ff7dd65d762/runners/image_editing.py
"""


import os

import numpy as np
import torch
from torchvision.utils import save_image

import argparse
import yaml

from model import Model

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_noise_schedule(config):
    betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
    num_time_steps = betas.shape[0]
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    model_var_type = config.model.var_type
    if model_var_type == "fixedlarge":
        logvar = np.log(np.append(posterior_variance[1], betas[1:]))

    elif model_var_type == 'fixedsmall':
        logvar = np.log(np.maximum(posterior_variance, 1e-20))

    alphas = torch.tensor(alphas)
    betas = torch.tensor(betas)
    alphas_cumprod = torch.tensor(alphas_cumprod)
    logvar = torch.tensor(logvar)

    return betas, alphas, alphas_cumprod, logvar, num_time_steps

def sample_stochastic_step(x, model, t, alpha_bar, beta, std):
    """
    Sample according to equation (3) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711

    TODO Ensure that our noise parameters match notations in the paper.
    Hypothesis: This is the same as VP SDE sampling used in the SDEdit code.
    """
    eps = model(x, t * torch.tensor([1.]))
    x = x - beta / (1 - alpha_bar) * eps
    x = x / (1 - beta) + std * torch.randn_like(x)
    return x