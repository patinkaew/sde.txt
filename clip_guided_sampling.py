"""
Guiding sampling by backpropagatin CLIP loss
"""

import os
import argparse

import torch
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
import clip
from tqdm import tqdm

import diffusion as diff
from model import Model
import util

def run_clip_guided_generation(args):
    
    # Inputs
    text = args.text

    # DiffusionCLIP parameters
    s_gen = args.s_gen  # Number of reverse ("generative") steps to go from x_t0 to x0
    lr = args.lr        # Adam optimizer initial learning rate
    nudge_iter = args.nudge_iter  # Number of iterations to finetune the image with CLIP

    # General generation parameters
    config_path = args.config
    ckpt_path = args.ckpt
    save_path = args.save_path
    log_every = args.log_every
    batch_size = args.batch
    
    # General setup
    torch.manual_seed(77) # 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    util.mkdir_if_not_exists(save_path)
    util.print_arguments(args)
    util.log_arguments(args, os.path.join(save_path, 'args.txt'))
    config = util.load_config(config_path)

    print('Device: {}'.format(device))
    print('Set up...')

    # Set up diffusion parameters
    alphas_cumprod, alphas_cumprod_prev = diff.get_clip_guided_schedule(config, s_gen, device)

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
    
    print('Begin nudging the generative process...')

    # Reverse x_T to x0
    ones = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, config.data.channels, 
                    config.data.image_size, config.data.image_size, device=device)
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.2)
    for i in tqdm(range(nudge_iter), total=nudge_iter):
        if not i % log_every:
            with torch.no_grad():
                util.save_image_batch(x, save_path, i)
        x_guided = x.clone()
        for t in reversed(range(s_gen)):
            x_guided = diff.sample_deterministic_step(x_guided, model, t, alphas_cumprod[t], alphas_cumprod_prev[t], ones)
        image_encoding = clip_model.encode_image(clip_preprocess(x_guided))
        clip_loss = 1. - F.cosine_similarity(image_encoding, text_encoding)
        clip_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
            
    util.save_image_batch(x, save_path, 'final')

def main():
    parser = argparse.ArgumentParser(description='CLIP-guided generation')
    parser.add_argument('--text', type=str)
    parser.add_argument('--s_gen', type=int, default=10)
    parser.add_argument('--nudge_iter', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config', type=str, default='config_yml/cifar10.yml')
    parser.add_argument('--ckpt', type=str, default='model_ckpt/cifar10.ckpt')
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=50)
    args = parser.parse_args()
    run_clip_guided_generation(args)

if __name__ == '__main__':
    main()
