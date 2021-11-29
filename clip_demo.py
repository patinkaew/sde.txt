"""
Playing around with CLIP
"""

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# for param in model.parameters():
#     print(param.requires_grad)  # Default to True
print(model.training)  # Default to eval

image = preprocess(Image.open("rieri.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["young woman"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)