# Text-guided Image Generation with Diffusion Model and CLIP

This repository contains the code associated with the project "Text-guided Image Generation with Diffusion Model and CLIP" by Kao Kitichotkul and Patin Inkaew. This project was done as a class project for CS 236: Deep Generative Model class in the Fall 2021 quarter at Stanford University.

## Requirements
We develop this project on Google Cloud Platform. We run our experiments with NVIDIA Tesla K80 GPU.
The code has been tested for Python version 3.7.12. The following are the required packages, together with the versions we have been tested on:
- numpy = 1.19.5
- matplotlib = 3.5.0
- pytorch = 1.10.0
- torchvision = 0.11.1+cu111
- Pillow = 8.4.0
- PyYAML = 6.0
- tqdm = 4.62.3
- [clip](https://github.com/openai/CLIP) = 1.0
- [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/) = 0.6.0
- [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/) = 0.3.0

We also provide `requirements.txt` file which can be used to install required packages with `pip` command.

## Abstract

## Content
Here are important files:


## Reference and Acknowledgement
If you find this repository useful for your research, please cite the following work,\

```
@misc{sde.txt,
      title={Text-guided Image Generation with Diffusion Model and CLIP}, 
      author={Ruangrawee Kitichotkul and Patin Inkaew},
      year={2021},
      note={CS 236 class project at Stanford University}
}
```

This experiment is based on/inspired by

### Inception Score (IS) and Kernel Inception Score (KID) Demo
We wrote this demo to test evaluation metrics (IS/KID). We use pretrain GAN models from [PyTorch Pretrained GANs](https://github.com/lukemelas/pytorch-pretrained-gans). If you wish to run this demo, you will need to install [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) since the pretrain GANs repository uses this library. Since we use models trained on ImageNet dataset, we reference label file from [imagenet 1000 class idx to human readable labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). For real image referenc, we use downloader from [ImageNet Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader)

### CIFAR10 
We use model checkpoint from DDPM model included in [PyTorch pretrained Diffusion Models](https://github.com/pesser/pytorch_diffusion) repository. We use config file, inclduing noise scheduling from [Denoising Diffusion Implicit Models (DDIM)](https://github.com/ermongroup/ddim). For real image reference, we use pytorch dataloader and PIL to convert to PNG images.

### CELEBA-HQ 256 x 256, LSUN Bedroom, LSUN Church
We use model chekpoints and config file from [SDEdit: Image Synthesis and Editing with Stochastic Differential Equations](https://github.com/ermongroup/SDEdit).

### Development
We wrote prototype codes in jupyter notebooks and we would like to thank the tutorial [Google Cloud Setup and Tutorial](https://github.com/cs231n/gcloud) from CS 231N: Convolutional Neural Networks for Visiual Recognition class at Stanford University.

### Acknowledgement
This project will not be possible without supports from many individuals. We would like to give special thank to our TA project mentor Kelly He for her guideline throughout the project development. We would like to thank all CS 236 instructors and staff for this opportunity,  thoughtful instructions, and great supports.

## Contact

Please contact Kao Kitichotkul (rkitichotkul at stanford dot edu) or Patin Inkaew (pinkaew at stanford dot edu) if you have any question.
