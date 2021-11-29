import os
import argparse
import yaml
import torch
from torchvision.utils import save_image

def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return config

def load_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

def save_image_batch(image_batch, save_path, t, save_tensor=False):
    batch_size = image_batch.shape[0]
    for i in range(batch_size):
        mkdir_if_not_exists(os.path.join(save_path, str(i)))
        save_image(image_batch[i], os.path.join(save_path, str(i), '{}.png'.format(t)))
        if save_tensor:
            torch.save(image_batch[i], os.path.join(save_path, str(i), '{}.pt'.format(t)))


def turn_off_model_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def print_arguments(args):
    for arg in vars(args):
        print('{}: {}'.format(arg, getattr(args, arg)))


def log_arguments(args, path):
    with open(path, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
