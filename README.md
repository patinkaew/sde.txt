# Text-guided Image Generation with Diffusion Model and CLIP
## Libraries
- torchmetrics
- torch-fidelity
- [PyTorch Pretrained GANs](https://github.com/lukemelas/pytorch-pretrained-gans) for loading pretrained GANs models to testing evaluation 
- boto3 (If using PyTorch Pretrained GANs repo)
## Reference and Acknowledgement

This experiment is based on/inspired by

## Inception Score (IS) and Kernel Inception Score (KID) Demo
We wrote this demo to test evaluation metrics (IS/KID). We use pretrain GAN models from [PyTorch Pretrained GANs](https://github.com/lukemelas/pytorch-pretrained-gans). Since we use models trained on ImageNet dataset, we reference label file from [imagenet 1000 class idx to human readable labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). For real image referenc, we use downloader from [ImageNet Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader)

## CIFAR10 
We use CIFAR10 model checkpoint from DDPM included in [PyTorch pretrained Diffusion Models](https://github.com/pesser/pytorch_diffusion) repository. We use config file, inclduing noise scheduling from [Denoising Diffusion Implicit Models (DDIM)](https://github.com/ermongroup/ddim). For real image reference, we use pytorch dataloader and PIL to convert to PNG images.

## CELEBA-HQ 256 x 256, LSUN
- https://github.com/ermongroup/SDEdit
