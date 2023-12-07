# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:54:47 2023

@author: 45157
"""

# get text feature
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor

model = CLIPModel.from_pretrained("models/clip/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("models/clip/clip-vit-base-patch16")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs).detach().numpy()

# get image feature
from PIL import Image
import requests

model = CLIPModel.from_pretrained("models/clip/clip-vit-base-patch16")
processor = AutoProcessor.from_pretrained("models/clip/clip-vit-base-patch16")

# image = Image.open("test/data/image/boy.jpeg")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt", padding=True)
image_features = model.get_image_features(**inputs).detach().numpy()

# get text feature
from transformers import AutoTokenizer, CLIPTextModel

model = CLIPTextModel.from_pretrained("models/clip/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("models/clip/clip-vit-base-patch16")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state.detach().numpy()
pooled_output = outputs.pooler_output.detach().numpy()  # pooled (EOS token) states



