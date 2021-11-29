"""
Ablation study of parameters for conditional generation by using CLIP to model p(y | x).
"""

import os
import argparse

import torch
from torchvision import transforms
import clip
from tqdm import tqdm

import diffusion as diff
from model import Model
import util

def run_cond_gen(args):
    
    # Inputs
    text = args.text

    # Generation parameters
    config_path = args.config
    ckpt_path = args.ckpt
    save_path = args.save_path
    batch_size = args.batch
    log_every = args.log_every
    sampling_method = args.sampling_method
    cond_scaling = args.cond_scaling
    time_guiding_start = args.guiding_start
    
    # General setup
    torch.manual_seed(236) # 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    util.mkdir_if_not_exists(save_path)
    util.print_arguments(args)
    util.log_arguments(args, os.path.join(save_path, 'args.txt'))
    config = util.load_config(config_path)

    print('Device: {}'.format(device))
    print('Set up...')

    # Set up diffusion parameters
    betas, alphas_cumprod, alphas_cumprod_prev, \
        logvar, num_time_steps = diff.get_noise_schedule(config, device)
    std = torch.exp(0.5 * logvar) # only for stochastic sampling
    ones = torch.ones(batch_size, device=device)

    # Load diffusion model
    model = Model(config)
    util.load_model(model, ckpt_path, device)
    util.turn_off_model_requires_grad(model)

    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/32")
    n_px = 224  #clip_model.input_resolution.item()
    clip_preprocess = transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    util.turn_off_model_requires_grad(clip_model)

    # Encode text
    text_feature = clip.tokenize([text]).to(device=device)
    text_encoding = clip_model.encode_text(text_feature).detach().clone()
    
    print('Begin sampling...')

    # Sampling
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in tqdm(reversed(range(num_time_steps)), total=num_time_steps):
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t)
        if t == time_guiding_start:
            print('t = {}, start CLIP guiding...'.format(t))
        if t > time_guiding_start:
            if sampling_method == 'sto':
                x = diff.sample_stochastic_step(x, model, t, alphas_cumprod[t], betas[t], std[t], ones)
            elif sampling_method == 'det':
                x = diff.sample_deterministic_step(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
            else:
                raise ValueError('Invalid sampling method')
        else:
            if sampling_method == 'sto':
                x = diff.sample_cond_stochastic_step(x, model, t, 
                        alphas_cumprod[t], betas[t], std[t], ones, 
                        clip_model, clip_preprocess, text_encoding, cond_scaling)
            elif sampling_method == 'det':
                x = diff.sample_cond_deterministic_step(x, model, t, 
                        alphas_cumprod[t], alphas_cumprod_prev[t], ones, 
                        clip_model, clip_preprocess, text_encoding, cond_scaling)
            else:
                raise ValueError('Invalid sampling method')
            
    util.save_image_batch(x, save_path, 'final')

def main():
    parser = argparse.ArgumentParser(description='Ablation study')
    parser.add_argument('--text', type=str)
    parser.add_argument('--cond_scaling', type=float, default=1000)
    parser.add_argument('--guiding_start', type=int, default=1000)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config', type=str, default='config_yml/celeba.yml')
    parser.add_argument('--ckpt', type=str, default='model_ckpt/celeba_hq.ckpt')
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--sampling_method', type=str, default='sto', choices=['sto']) # TODO add det later
    args = parser.parse_args()
    run_cond_gen(args)

if __name__ == '__main__':
    main()
