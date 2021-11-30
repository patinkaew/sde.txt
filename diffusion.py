"""
Sampling with Diffusion model

Adapted from
    https://github.com/ermongroup/SDEdit/blob/c0ed910a759df68ecc373caa020f6ff7dd65d762/runners/image_editing.py
"""

import numpy as np
import torch
from torch.nn import functional as F

"""Sampling methods"""

def sample_stochastic_step(x, model, t, alpha_bar, beta, std, ones):
    """
    Sample according to equation (3) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711

    Note: This is almost identical to VP SDE sampling used in the SDEdit code.
        https://github.com/ermongroup/SDEdit/blob/c0ed910a759df68ecc373caa020f6ff7dd65d762/runners/image_editing.py#L30
        Except there is masking in SDEdit.

    Note 2: VP SDE is equivalent to DDPM sampling in Ho et al. 2020.
    """
    eps = model(x, t * ones)
    mean = x - beta / (1 - alpha_bar).sqrt() * eps
    x = mean / (1 - beta).sqrt() + std * torch.randn_like(x)
    return x

def sample_deterministic_step(x, model, t, alpha_bar, alpha_bar_prev, ones):
    """
    Sample according to equation (13) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711

    Note: This is the DDIM sampling procedure.
        https://arxiv.org/abs/2010.02502
    """
    eps = model(x, t * ones)
    mean = x - (1 - alpha_bar).sqrt() * eps
    mean = (alpha_bar_prev / alpha_bar).sqrt() * mean
    x = mean + (1 - alpha_bar_prev).sqrt() * eps
    return x

def invert_deterministic_step(x, model, t, alpha_bar, alpha_bar_next, ones):
    """
    Invert image to noise according to equation (12) in DiffusionCLIP.
        https://arxiv.org/abs/2110.02711
    """
    eps = model(x, t * ones)
    mean = x - (1 - alpha_bar).sqrt() * eps
    mean = (alpha_bar_next / alpha_bar).sqrt() * mean
    x = mean + (1 - alpha_bar_next).sqrt() * eps
    return x

def sample_cond_stochastic_step(x, model, t, alpha_bar, beta, std, ones, clip_model, clip_preprocess, text_features, cond_scaling, print_clip=False):
    """
    Sampling like in sample_stochastic_step, but with CLIP to model log p(y|x)
    """
    # Calculate the conditional term - modeling grad log p(y | x)
    x.requires_grad = True
    #image_encoding = clip_model.encode_image(clip_preprocess(x))
    #cond_term = torch.log(F.cosine_similarity(image_encoding, text_encoding))
    image = clip_preprocess(x)
    logits_per_image, logits_per_text = clip_model(image, text_features)
    probs_per_image = logits_per_image.softmax(dim=-1)
    cond_term = torch.log(probs_per_image[:, 0]) # target text will always be the first
    cond_term_grad = torch.autograd.grad(cond_term, x, torch.ones_like(cond_term))[0].detach()
    x.requires_grad = False
    # print('cond_term_grad', cond_term_grad.shape, cond_term_grad.mean(), cond_term_grad.std(), cond_term_grad.requires_grad)
    clip_prob = probs_per_image.cpu().detach().numpy()
    if print_clip:
        print('probs: ', clip_prob)
        print('target probs: ', clip_prob[:, 0])
        #print('cosine similarity', cond_term)

    # Your usual reverse DDPM diffusion step
    eps = model(x, t * ones)
    # print('eps', eps.shape, eps.mean(), eps.std(), eps.requires_grad)
    mean = x - beta / (1 - alpha_bar).sqrt() * eps + cond_scaling * beta * cond_term_grad
    x = mean / (1 - beta).sqrt() + std * torch.randn_like(x)
    return x, clip_prob

def sample_cond_deterministic_step(x, model, t, alpha_bar, alpha_bar_prev, ones, clip_model, clip_preprocess, text_encoding, cond_scaling):
    """
    Sampling like in sample_deterministic_step, but with CLIP to model log p(y|x)
    TODO: change conditioning to log cossim approach.
    """
    # Calculate the conditional term - modeling grad log p(y | x)
    x.requires_grad = True
    image_encoding = clip_model.encode_image(clip_preprocess(x))
    cond_term = F.cosine_similarity(image_encoding, text_encoding)
    cond_term_grad = torch.autograd.grad(cond_term, x).detach()
    x.requires_grad = False

    # Your usual reverse DDIM diffusion step
    eps = model(x, t * ones)
    mean = x - (1 - alpha_bar).sqrt() * (eps - cond_scaling * cond_term_grad)
    mean = (alpha_bar_prev / alpha_bar).sqrt() * mean
    x = mean + (1 - alpha_bar_prev).sqrt() * eps
    return x

"""Setting up noise variance schedules"""

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_noise_schedule(config, device):
    """
    This noise schedule is for models from SDEdit.
    """
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

def get_diffusion_clip_schedule(config, t0, s_inv, s_gen, device):
    betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
    betas = betas[t0:]

    # Get forward ("inverse") schedule
    betas_inv = get_beta_schedule(
            beta_start=betas[0],
            beta_end=betas[-1],
            num_diffusion_timesteps=s_inv
        )
    alphas_inv = 1.0 - betas_inv
    alphas_cumprod_inv = np.cumprod(alphas_inv, axis=0)
    alphas_cumprod_next_inv = np.append(alphas_cumprod_inv[1:], 1.0)

    # Get reverse ("generative") schedule
    betas_gen = get_beta_schedule(
            beta_start=betas[0],
            beta_end=betas[-1],
            num_diffusion_timesteps=s_gen
        )
    alphas_gen = 1.0 - betas_gen
    alphas_cumprod_gen = np.cumprod(alphas_gen, axis=0)
    alphas_cumprod_prev_gen = np.append(1.0, alphas_cumprod_gen[:-1])

    alphas_cumprod_inv = torch.tensor(alphas_cumprod_inv).to(device)
    alphas_cumprod_next_inv = torch.tensor(alphas_cumprod_next_inv).to(device)
    alphas_cumprod_gen = torch.tensor(alphas_cumprod_gen).to(device)
    alphas_cumprod_prev_gen = torch.tensor(alphas_cumprod_prev_gen).to(device)

    return alphas_cumprod_inv, alphas_cumprod_next_inv, alphas_cumprod_gen, alphas_cumprod_prev_gen