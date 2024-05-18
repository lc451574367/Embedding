# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:00:54 2024
    
Module Help:
load model : 
	get_gensim_model
	gloveTransferW2v
	get_bert_model
    get_gpt2_model
    get_clip_model
    get_elmo_model
    get_vgg_model
    
@author: ChengLiu
"""

import numpy as np
import os, signal

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer, CLIPModel, AutoProcessor,AutoTokenizer
import torch
from PIL import Image
from allennlp.modules.elmo import Elmo, batch_to_ids
import torchvision.models as models

from feature.model_path import *
from preprocessing.textprocess import *


def get_gensim_model(modelpath,binary):
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(modelpath), binary=binary)
        return model
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None    

def gloveTransferW2v(modelpath):
    f,_ = os.path.splitext(modelpath) 
    model_g2w = 'models/glove/' + os.path.basename(f) + '.g2w.txt'
    glove2word2vec(modelpath, model_g2w)
    
def get_glove_model(modelpath):
    f,_ = os.path.splitext(modelpath)  
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(modelpath), binary=False)
    except:
        if 'g2w' in f.split('.'):
            print("We can't visit models, please check your model path")
        else:
            print('Transfer glove format to word2vec format ...')
            gloveTransferW2v(modelpath)
            
    return model

def get_bert_model(modelpath):
    try:
        tokenizer = BertTokenizer.from_pretrained(modelpath)
        model = BertModel.from_pretrained(modelpath)
        return model,tokenizer
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None

def get_gpt2_model(modelpath):
    try :
        model = GPT2Model.from_pretrained(modelpath)
        tokenizer = GPT2Tokenizer.from_pretrained(modelpath)
        return model,tokenizer
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None   

def get_clip_model(modelpath):
    try:
        model = CLIPModel.from_pretrained(modelpath)
        processor = AutoProcessor.from_pretrained(modelpath)
        tokenizer = AutoTokenizer.from_pretrained(modelpath)
        return model,processor,tokenizer
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None

def get_elmo_model(modelpath):
    f,_ = os.path.splitext(modelpath)  
    jsonpath = f.replace("weights","options") + '.json'
    try:
        # Initialize ELMO model
        model = Elmo(jsonpath,modelpath,num_output_representations=1, dropout=0)
        return model
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None

def get_vgg_model(modelpath):
    try :
        model = models.vgg19(pretrained=False)
        pretrained_state_dict = torch.load(modelpath)
        model.load_state_dict(pretrained_state_dict)
        
        # 添加一个展平层
        flatten = torch.nn.Flatten()
        # 新的分类器部分
        new_classifier = torch.nn.Sequential(flatten, torch.nn.Linear(512 * 7 * 7, 4096))        
        # 替换vgg19的分类器部分
        model.classifier = new_classifier
        model = model.eval()
        return model
    except Exception as e:
        print('\n------------------------------------')
        print("Error loading model: {}".format(e))
        print('------------------------------------\n')
        return None
    
                