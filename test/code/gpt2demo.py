# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:47:09 2023

@author: 45157
"""

from transformers import GPT2Model, GPT2Tokenizer
import torch

gpt2_path = 'models/gpt2'

model = GPT2Model.from_pretrained(gpt2_path)
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)

input_text = "Your input text here."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model(input_ids)
    
embedding = output.last_hidden_state.numpy().squeeze()


    
