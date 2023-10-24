# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:28:23 2023

@author: Cheng Liu
"""
from model_path import *
from get_vector import *
from textprocess import *
from gensim.scripts.glove2word2vec import glove2word2vec

"Transfer format"
"""
If you download the glove pre-trained vector from the official website,
you should first convert the format to word2vec, then you will use the glove vector after format coversion to get embedding vectors
"""
glove_model = 'models/glove/glove.6B.50d.txt'
g2w_glove_model = 'models/glove/glove.6B.50d.g2w.txt'
glove2word2vec(glove_model, g2w_glove_model)

"1. word embedding vector"
wordpath = 'test/test.txt'
Word_vector(wordpath, modeltype='glove', corpus='common', dimension=300)
Word_vector(wordpath, modeltype='word2vec', corpus='GoogleNews')

"2. sentence embedding vector"
sentencepath = 'test/sentence.txt'
Sentence_vector(sentencepath, modeltype = 'bert', layer = 12, dimension=768) 
Sentence_vector(sentencepath, modeltype = 'glove')
Sentence_vector(sentencepath, modeltype = 'word2vec')
