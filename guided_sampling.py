"""
Guiding DDPM sampling using CLIP
"""

import torch
import clip
from PIL import Image

def clip_loss(clip_model, image_embedding, text_embedding):
    pass