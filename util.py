import os
import argparse
import yaml
import torch
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

to_tensor = transforms.ToTensor()

def load_image(path):
    return to_tensor(Image.open(path)).float()

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


def save_clip_prob(clip_prob_hist, texts, save_path):
    plt.clf()
    num_cond_time_step, num_texts = clip_prob_hist.shape
    for text_id in range(num_texts):
        plt.plot(range(num_cond_time_step), clip_prob_hist[:, text_id], label=texts[text_id])

    plt.xlabel("conditional time steps")
    plt.ylabel("CLIP probabilities")
    plt.title("CLIP probabilities over conditional time steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "/clip_prob.png", bbox_inches = "tight")


def save_clip_prob_batch(clip_prob_hist, texts, save_path):
    num_cond_time_step, batch_size, num_texts = clip_prob_hist.shape
    assert len(texts) == num_texts
    if num_cond_time_step == 0:
        return
    for i in range(batch_size):
        save_clip_prob(clip_prob_hist[:, i, :], texts, save_path + "/{}".format(i))
