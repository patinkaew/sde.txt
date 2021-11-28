"""
Conditional sampling by using CLIP to model log p(y | x)
"""

import torch
from torchvision import transforms
import clip
from tqdm import tqdm

import diffusion as diff
from model import Model
import util

def main():
    
    # Inputs
    text = "old woman"

    # Generation parameters
    config_path = 'config_yml/celeba.yml'
    ckpt_path = 'model_ckpt/celeba_hq.ckpt'
    save_path = 'result/condgen-celeba-0'
    batch_size = 3
    log_every = 50
    sampling_method = 'sto'
    cond_scaling = 1e3  # To sweep

    # General setup
    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    util.mkdir_if_not_exists(save_path)
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

    # Sampling
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    for t in tqdm(reversed(range(num_time_steps)), total=num_time_steps):
        if not t % log_every:
            print('Time step {}'.format(t))
            util.save_image_batch(x, save_path, t)
        if sampling_method == 'sto':
            x = diff.sample_cond_stochastic_step(x, model, t, 
                    alphas_cumprod[t], betas[t], std[t], ones, 
                    clip_model, clip_preprocess, text_encoding, cond_scaling, not t % log_every)
        elif sampling_method == 'det':
            x = diff.sample_cond_deterministic_step(x, model, t, 
                    alphas_cumprod[t], alphas_cumprod_prev[t], ones, 
                    clip_model, clip_preprocess, text_encoding, cond_scaling)
        else:
            raise ValueError('Invalid sampling method')

if __name__ == '__main__':
    main()
