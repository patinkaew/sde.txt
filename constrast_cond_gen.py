"""
Constrastive study of parameters for conditional generation by using CLIP to model p(y | x).
"""

import os
import argparse

import numpy as np
import torch
from torchvision import transforms
import clip
from tqdm import tqdm

import diffusion as diff
from model import Model
import util

def run_cond_gen(args):
    
    # Inputs
    #text = args.text

    # Generation parameters
    config_path = args.config
    ckpt_path = args.ckpt
    save_path = args.save_path
    batch_size = args.batch
    log_every = args.log_every
    sampling_method = args.sampling_method
    cond_scaling = args.cond_scaling
    time_guiding_start = args.guiding_start
    cond_type = args.cond_type
    
    # Seed parameters
    seed = args.seed
    use_seed = args.use_seed
    
    # General setup
    if use_seed:
        torch.manual_seed(seed) #0
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
    #text_feature = clip.tokenize([text]).to(device=device)
    #text_encoding = clip_model.encode_text(text_feature).detach().clone()
    
    # Process text prompts
    print('Process text prompts...')
    target_text = args.target_text
    constrast_texts = args.constrast_texts
    texts = [target_text]
    if len(constrast_texts) > 0:
        constrast_texts_list = [each_text.strip() for each_text in constrast_texts.split(',')]
        texts += constrast_texts_list
    text_features = clip.tokenize(texts).to(device)
    print('target_text: {} constrast_texts: {}'.format(texts[0], texts[1:]))
    
    print('Begin sampling...')
    clip_prob_hist = np.zeros((0, batch_size, len(texts)))

    # Sampling
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in tqdm(reversed(range(num_time_steps)), total=num_time_steps):
        #if t == num_time_steps -2:
        #    return 0
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t)
        if t > time_guiding_start:
            if sampling_method == 'sto':
                x = diff.sample_stochastic_step(x, model, t, alphas_cumprod[t], betas[t], std[t], ones)
            elif sampling_method == 'det':
                x = diff.sample_deterministic_step(x, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
            else:
                raise ValueError('Invalid sampling method')
        else:
            if t + 1 == time_guiding_start:
                print('t+1 = {}, start CLIP guiding...'.format(t+1))
            if sampling_method == 'sto':
                x, clip_prob = diff.sample_cond_stochastic_step(x, model, t, 
                        alphas_cumprod[t], betas[t], std[t], ones, 
                        clip_model, clip_preprocess, text_features, cond_scaling, cond_type)
            elif sampling_method == 'det':
                x = diff.sample_cond_deterministic_step(x, model, t, 
                        alphas_cumprod[t], alphas_cumprod_prev[t], ones, 
                        clip_model, clip_preprocess, text_features, cond_scaling)
            else:
                raise ValueError('Invalid sampling method')
            clip_prob_hist = np.concatenate([clip_prob_hist, np.expand_dims(clip_prob, axis=0)], axis=0)
      
    util.save_image_batch(x, save_path, 'final')
    util.save_clip_prob_batch(clip_prob_hist, texts, save_path) 

def main():
    parser = argparse.ArgumentParser(description='Contrastive study')
    parser.add_argument('--seed', type=int, default=236)
    parser.add_argument('--use_seed', type=bool, default=1)
    parser.add_argument('--target_text', type=str)
    parser.add_argument('--constrast_texts', type=str, default='')
    parser.add_argument('--cond_scaling', type=float, default=1000)
    parser.add_argument('--guiding_start', type=int, default=1000)
    parser.add_argument('--cond_type', type=str, default='contrastive')
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
