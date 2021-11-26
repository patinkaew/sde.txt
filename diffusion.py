"""
Sampling with Diffusion model

Adapted from
    https://github.com/ermongroup/SDEdit/blob/c0ed910a759df68ecc373caa020f6ff7dd65d762/runners/image_editing.py
"""


import numpy as np
import torch

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_noise_schedule(config, device):
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

    betas = torch.tensor(betas).to(device=device)
    alphas_cumprod = torch.tensor(alphas_cumprod).to(device=device)
    alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev).to(device=device)
    logvar = torch.tensor(logvar).to(device=device)

    return betas, alphas_cumprod, alphas_cumprod_prev, logvar, num_time_steps

def sample_stochastic_step(x, model, t, alpha_bar, alpha_bar_prev, beta, std, ones):
    """
    Sample according to equation (3) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711

    Note: This is almost identical to VP SDE sampling used in the SDEdit code.
        https://github.com/ermongroup/SDEdit/blob/c0ed910a759df68ecc373caa020f6ff7dd65d762/runners/image_editing.py#L30

    TODO Ensure that our noise parameters match notations in the paper.
    """
    eps = model(x, t * ones)
    mean = x - beta / (1 - alpha_bar).sqrt() * eps
    x = mean / (1 - beta).sqrt() + std * torch.randn_like(x)
    return x

def sample_deterministic_step(x, model, t, alpha_bar, alpha_bar_prev, beta, std, ones):
    """
    Sample according to equation (5) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711

    Note: This is the DDIM sampling procedure.
        https://arxiv.org/abs/2010.02502
    """
    eps = model(x, t * ones)
    mean = x - (1 - alpha_bar).sqrt() * eps
    mean = (alpha_bar_prev / alpha_bar).sqrt() * mean
    x = mean + (1 - alpha_bar_prev).sqrt() * eps
    return x
