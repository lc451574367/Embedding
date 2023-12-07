# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:34:52 2023

@author: 45157
"""

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
models_name = clip.available_models()

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image).softmax(dim=-1).cpu().numpy()
    text_features = model.encode_text(text).softmax(dim=-1).cpu().numpy()
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


image = preprocess(Image.open("test/data/image/boy.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["a boy with black hair"]).to(device)


outputs = model(image, text)
print(outputs)
