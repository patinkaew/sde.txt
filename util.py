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