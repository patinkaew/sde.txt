import os
import torch
from torch.utils.data import DataLoader
import torchvision
import util

to_tensor = torchvision.transforms.ToTensor()
to_PIL = torchvision.transforms.ToPILImage()

def main():

    # Arguments
    num_images_to_save = 20
    savedir = 'cifar10-images'

    util.mkdir_if_not_exists(savedir)
    dataset = torchvision.datasets.CIFAR10('cifar10-dataset', transform=to_tensor, download=True)
    dataloader = DataLoader(dataset)
    for i, (image, label) in enumerate(dataloader):
        if i >= num_images_to_save: break
        util.save_image(image, os.path.join(savedir, '{}.png'.format(i)))
        torch.save(image, os.path.join(savedir, '{}.pt'.format(i)))
        

if __name__ == '__main__':
    main()