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

# +
def run_diffusionclip(args):
    
    # Inputs
    text = args.text
    image_path = args.image

    # DiffusionCLIP parameters
    t0 = args.t0        # t0 indicating the noise level to forward to
    s_inv = args.s_inv  # Number of forward ("inverse") steps to go from x_0 to x_t0
    s_gen = args.s_gen  # Number of reverse ("generative") steps to go from x_t0 to x0
    lr = args.lr        # Adam optimizer initial learning rate
    nudge_iter = args.nudge_iter  # Number of iterations to finetune the image with CLIP
    lambda_id = args.id_weight

    # General generation parameters
    config_path = args.config
    ckpt_path = args.ckpt
    save_path = args.save_path
    log_every = args.log_every
    
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
    alphas_cumprod_inv, alphas_cumprod_next_inv, \
        alphas_cumprod_gen, alphas_cumprod_prev_gen = diff.get_diffusion_clip_schedule(config, t0, s_inv, s_gen, device)

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
    
    print('Begin inverting image x0 to noise x_t0...')

    ones = torch.ones(1, device=device)
    x = util.load_image(image_path).unsqueeze(0).to(device) # (1, C, H, W) shape image
    x0 = x.detach().clone()
    
    # Forward x_0 to x_t0
    for t in range(s_inv):
        x = diff.invert_deterministic_step(x, model, t, alphas_cumprod_inv[t], alphas_cumprod_next_inv[t], ones)
    util.save_image_batch(x, save_path, 'x_t0')
    
#     x = torch.randn(1, config.data.channels, 
#                     config.data.image_size, config.data.image_size, device=device)
#     for t in reversed(range(s_gen)):
#         x = diff.sample_deterministic_step(x, model, t, alphas_cumprod_gen[t], alphas_cumprod_prev_gen[t], ones)
#     util.save_image_batch(x, save_path, 'x0_recovered')
#     raise NotImplementedError

    print('Begin nudging the generative process...')

    # Reverse x_t0 to x0
    x.requires_grad = True
    optimizer = optim.Adam([x], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.2)
    for i in tqdm(range(nudge_iter), total=nudge_iter):
        if not i % log_every:
            with torch.no_grad():
                util.save_image_batch(x, save_path, i)
        x_guided = x.clone()
        for t in reversed(range(s_gen)):
            x_guided = diff.sample_deterministic_step(x_guided, model, t, alphas_cumprod_gen[t], alphas_cumprod_prev_gen[t], ones)
        image_encoding = clip_model.encode_image(clip_preprocess(x_guided))
        clip_loss = 1. - F.cosine_similarity(image_encoding, text_encoding)
        # clip_loss.backward()
        identity_loss = lambda_id * (x_guided - x0).abs().mean()
        loss = clip_loss + identity_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
            
    util.save_image_batch(x, save_path, 'final')


# -

def main():
    parser = argparse.ArgumentParser(description='DiffusionCLIP')
    parser.add_argument('--text', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--t0', type=int, default=400)
    parser.add_argument('--s_inv', type=int, default=40)
    parser.add_argument('--s_gen', type=int, default=6)
    parser.add_argument('--nudge_iter', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--id_weight', type=float, default=0.5)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config', type=str, default='config_yml/celeba.yml')
    parser.add_argument('--ckpt', type=str, default='model_ckpt/celeba_hq.ckpt')
    # parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=50)
    args = parser.parse_args()
    run_diffusionclip(args)

if __name__ == '__main__':
    main()
