import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torchvision
from torchmetrics import KID
import glob
from PIL import Image
import argparse
import util

to_tensor = torchvision.transforms.ToTensor()
to_PIL = torchvision.transforms.ToPILImage()

label_names = 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(', ')


def download_cifar10(num_images, targets, img_dir):

    count = np.zeros(len(targets))
    
    util.mkdir_if_not_exists(img_dir)
    dataset = torchvision.datasets.CIFAR10('cifar10-dataset', transform=to_tensor, download=True)
    dataloader = DataLoader(dataset)
    for i, (image, label) in enumerate(dataloader):
        label = int(label.cpu().detach().numpy())
        if label in targets and count[label] < num_images:
            save_dir = img_dir + '/{}'.format(label_names[label])
            util.mkdir_if_not_exists(save_dir)
            util.save_image(image, os.path.join(save_dir, '{}.png'.format(i)))
            #torch.save(image, os.path.join(savedir, '{}.pt'.format(i)))
            count[label] += 1
        if count.all() >= num_images:
            break


def load_images(img_dir, num_images):
    trans = T.Compose([T.ToTensor(), lambda x: x*255])
    img_list = []
    for filename in glob.glob(img_dir + "/*.png"):
        im = Image.open(filename)
        img_list.append(trans(im))
        if len(img_list) >= num_images:
            break
    return torch.stack(img_list).type(torch.ByteTensor)


def run_KID_eval(args):
    mode = args.mode
    num_images = args.num_images
    targets = args.targets
    generate_img_dir = args.generate_img_dir
    real_img_dir = args.real_img_dir
    subset_size = args.subset_size
    
    if mode == 'download':
        assert num_images > 0
        assert len(real_img_dir) > 0
        if len(targets) == 0:
            targets = np.arange(10)
        else:
            targets = [int(x.strip()) for x in targets.split(',')]
        print("Labels: {}".format([label_names[idx] for idx in targets]))
        
        print("Begin downloading...")
        download_cifar10(num_images, targets, real_img_dir)
        
    elif mode == 'eval':
        assert num_images > 0
        assert subset_size > 0
        assert num_images >= subset_size
        assert len(real_img_dir) > 0
        assert len(generate_img_dir) > 0
        kid = KID(subset_size=subset_size)
        kid.reset()
        real_rgb = load_images(real_img_dir, num_images)
        fake_rgb = load_images(generate_img_dir, num_images)
        print('Updating KID with real images...')
        kid.update(real_rgb, real=True)
        print('Updating KID with generated images...')
        kid.update(fake_rgb, real=False)
        print('KID score: {}'.format(kid.compute()))

    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='CIFAR10 KID evaluation')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--num_images', type=int, default=0)
    parser.add_argument('--targets', type=str, default='')
    parser.add_argument('--generate_img_dir', type=str, default='')
    parser.add_argument('--real_img_dir', type=str, default='')
    parser.add_argument('--subset_size', type=int, default=1000)
    args = parser.parse_args()
    run_KID_eval(args)


if __name__ == '__main__':
    main()
