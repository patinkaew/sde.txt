import os
import argparse
import yaml
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