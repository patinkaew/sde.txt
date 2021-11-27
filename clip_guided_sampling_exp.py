"""
Sampling from DDPM with stochastic inverse process.
"""

import os

import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
import clip

import diffusion as diff
from model import Model
import util

def main():
    
    # Arguments
    text = "tanned face"
    pinned_time_step = 50
    num_guiding_steps = 1000
    guide_lr = 1e-5
    config_path = 'config_yml/celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    save_path = 'result/cond-celeba-det-0'
    batch_size = 1
    log_every = 50
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sampler = diff.sample_deterministic_step

    print('Device: {}'.format(device))
    print('Set up...')

    util.mkdir_if_not_exists(save_path)

    # Set up parameters
    config = util.load_config(config_path)
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, device)
    std = torch.exp(0.5 * logvar)
    ones = torch.ones(batch_size, device=device)
    
    # Load diffusion model
    model = Model(config)
    util.load_model(model, ckpt_path, device)
    util.turn_off_model_requires_grad(model)

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device) # TODO should we use image preprocess?
    text_feature = clip.tokenize([text]).to(device=device)
    text_encoding = clip_model.encode_text(text_feature)

    print('Start sampling (no guiding yet)...')

    # Sampling
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in reversed(range(pinned_time_step, num_time_steps)):
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t, save_tensor=True)
        x = sampler(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        
    util.save_image_batch(x, save_path, 'before_guided', save_tensor=True)
    
    print('Start guiding generation with CLIP...')
        
    # Guiding setup
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=guide_lr)
    
    # Sampling and guiding
    for i in range(num_guiding_steps):
        print('guiding step', i)
        x_guided = x.clone()
        for t in reversed(range(pinned_time_step)):
            x_guided = sampler(x_guided, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        image_encoding = clip_model.encode_image(x_guided)
        clip_loss = 1 - F.cosine_similarity(image_encoding, text_encoding)
        clip_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    util.save_image_batch(x_guided, save_path, 'final', save_tensor=True)

def main2():
    """Guided generation starting from a saved time step"""
    
    # Arguments
    text = "tanned face"
    pinned_time_step = 50
    image_load_path = 'result/cond-celeba-det-0/0/{}.pt'.format(pinned_time_step)
    num_guiding_steps = 100
    guide_lr = 1e-5
    config_path = 'config_yml/celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sampler = diff.sample_deterministic_step
    config = util.load_config(config_path)
    
    print('Setting up...')
    
    # Load diffusion model
    model = Model(config)
    util.load_model(model, ckpt_path, device)
    util.turn_off_model_requires_grad(model)
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device) # TODO should we use image preprocess?
    text_feature = clip.tokenize([text]).to(device=device)
    text_encoding = clip_model.encode_text(text_feature)
    
    # Set up parameters
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, device)
    std = torch.exp(0.5 * logvar)
    ones = torch.tensor([1.], device=device)
    
    # Load saved noisy image
    x = torch.load(image_load_path).unsqueeze(0)
    
    # Guiding setup
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=guide_lr)
    
    print('Start guided generation...')
    
    # Sampling and guiding
    for i in range(num_guiding_steps):
        print('guiding step', i)
        x_guided = x.clone()
        for t in reversed(range(pinned_time_step)):
            print(t)
            x_guided = sampler(x_guided, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        image_encoding = clip_model.encode_image(x_guided)
        clip_loss = 1 - F.cosine_similarity(image_encoding, text_encoding)
        print('outhere!')
        clip_loss.backward()
        print('backward')
        optimizer.step()
        optimizer.zero_grad()
        
    util.save_image_batch(x_guided, save_path, 'final', save_tensor=True)


def main3():
    """Guided generation starting from a saved time step
    
    Try larger step size
    """
    
    # Arguments
    text = "tanned face"
    compressed_reverse_steps = 20
    pinned_time_step = 50
    image_load_path = 'result/cond-celeba-det-0/0/{}.pt'.format(pinned_time_step)
    num_guiding_steps = 100
    guide_lr = 1e-5
    config_path = 'config_yml/celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sampler = diff.sample_deterministic_step
    config = util.load_config(config_path)
    
    print('Setting up...')
    
     # Set up parameters
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, torch.device('cpu'))
    ones = torch.tensor([1.], device=device)
    
    alphas_cumprod, alphas_cumprod_prev, num_time_steps \
        = get_noise_schedule_manual(betas[0].item(), betas[pinned_time_step].item(), compressed_reverse_steps, device)
    
    print(alphas_cumprod)
    
    # Load diffusion model
    model = Model(config)
    util.load_model(model, ckpt_path, device)
    util.turn_off_model_requires_grad(model)
    
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device) # TODO should we use image preprocess?
    text_feature = clip.tokenize([text]).to(device=device)
    text_encoding = clip_model.encode_text(text_feature)
    
    
    # Load saved noisy image
    x = torch.load(image_load_path).unsqueeze(0)
    
    for t in reversed(range(compressed_reverse_steps)):
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t, save_tensor=True)
        x = sampler(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
    
    raise NotImplementedError
    
    # Guiding setup
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=guide_lr)
    
    print('Start guided generation...')
    
    # Sampling and guiding
    for i in range(num_guiding_steps):
        print('guiding step', i)
        x_guided = x.clone()
        for t in reversed(range(pinned_time_step)):
            print(t)
            x_guided = sampler(x_guided, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        image_encoding = clip_model.encode_image(x_guided)
        clip_loss = 1 - F.cosine_similarity(image_encoding, text_encoding)
        print('outhere!')
        clip_loss.backward()
        print('backward')
        optimizer.step()
        optimizer.zero_grad()
        
    util.save_image_batch(x_guided, save_path, 'final', save_tensor=True)

if __name__ == '__main__':
    main3()
