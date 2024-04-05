# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:28:23 2023

@author: Cheng Liu
"""
from model_path import *
from get_vector import *
from textprocess import *
from gensim.scripts.glove2word2vec import glove2word2vec

"1. Download pre-trained models"
"""
Glove: https://nlp.stanford.edu/projects/glove/glove.xx.xx.txt
e.g. https://nlp.stanford.edu/projects/glove/glove.42B.300d.txt

word2vec: https://code.google.com/archive/p/word2vec/

bert: https://huggingface.co/google/bert_xx_L-xx_H-xx_A-xx
e.g. https://huggingface.co/google/bert_uncased_L-12_H-768_A-12

gpt2: https://huggingface.co/gpt2

clip: https://huggingface.co/openai/clip-vit-base-patchxx
e.g. https://huggingface.co/openai/clip-vit-base-patch32

Please download models from the official website above, if you want to try more parameters of some models, you must download the model that corresponds to them 
"""

"2. Transfer format"
"""
If you download the glove pre-trained vector from the official website,
you should first convert the format to word2vec, then you can use the glove vector after format coversion to get embedding vectors
"""
glove_model = 'models/glove/glove.6B.50d.txt'
g2w_glove_model = 'models/glove/glove.6B.50d.g2w.txt'
glove2word2vec(glove_model, g2w_glove_model)

"3. Get embedding vector"
"""
After download the models, you can get different embedding vectors by different parameters.
"""
"3.1 get word embedding"
wordpath = 'test/data/text/test.txt'
Text_vector(wordpath, modeltype='glove', corpus='common', dimension=300)
Text_vector(wordpath, modeltype='word2vec', corpus='GoogleNews')
"3.2 get sentence embedding"
sentencepath = 'test/data/text/sentence.txt'
Text_vector(sentencepath, modeltype = 'bert') 
Text_vector(sentencepath, modeltype = 'glove')
Text_vector(sentencepath, modeltype = 'word2vec')
Text_vector(sentencepath, modeltype = 'gpt2')
Text_vector(sentencepath, modeltype = 'clip')
"3.3 get image embedding"
imagepath = 'test/data/image/imagefile.xlsx'
Image_vector(imagepath)
"3.4 get chinese word or sentence embedding"
chinesepath = 'test/data/text/chinese.txt'
Text_vector(chinesepath, modeltype = 'gpt2')
Text_vector(chinesepath, modeltype = 'bert',language='cn')
