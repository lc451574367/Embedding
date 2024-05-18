# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:47:27 2024

@author: 45157
"""

from feature.get_vector import *
from preprocessing.textprocess import *
from gensim.scripts.glove2word2vec import glove2word2vec


"2. Get embedding vector"
"""
After download the models, you can get different embedding vectors by different model labels.
"""
"2.1 get word embedding"
wordpath = '../test/data/text/test.txt'

Vector = Text_vector(wordpath, vectorType='word', label = 'w2v_base')  # 测试1
Vector = Text_vector(wordpath, vectorType='word', label = 'glv_base') 
Vector = Text_vector(wordpath, vectorType='word', label = 'fast_base') 
Vector = Text_vector(wordpath, vectorType='word', label = 'rws_base') 
Vector = Text_vector(wordpath, vectorType='word', label = 'cnt_base') 
Vector = Text_vector(wordpath, vectorType='word', outword='y') # default model is gpt2_base # 模型不合适做单独的词向量 # 测试2
Vector = Text_vector(wordpath, vectorType='word', label = 'gpt2_medium')  # 模型不合适做单独的词向量
Vector = Text_vector(wordpath, vectorType='word', label = 'gpt2_large')  # 模型不合适做单独的词向量
Vector = Text_vector(wordpath, vectorType='word', label = 'gpt2_xl')  # 模型不合适做单独的词向量
Vector = Text_vector(wordpath, vectorType='word', label = 'clip_base')  # 模型不合适做单独的词向量 # 测试3
Vector = Text_vector(wordpath, vectorType='word', label = 'bert_base') # 模型不合适做单独的词向量  # 测试4
Vector = Text_vector(wordpath, vectorType='word', label = 'elmo_base')  # 模型不合适做单独的词向量 # 测试5


"2.2 get sentence embedding"
sentencepath = '../test/data/text/sentence.txt'
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'w2v_base')  # 模型不合适做句子级别词向量 # 测试1
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'glv_base') # 模型不合适做句子级别词向量
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'fast_base') # 模型不合适做句子级别词向量
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'rws_base') # 模型不合适做句子级别词向量
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'cnt_base') # 模型不合适做句子级别词向量
Vector = Text_vector(sentencepath, vectorType='sentence') # default model is gpt2_base  # 测试2
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'gpt2_medium')  
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'gpt2_large')   
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'gpt2_xl')   
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'clip_base')  # 测试3
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'bert_base')  # 测试4
Vector = Text_vector(sentencepath, vectorType='sentence', label = 'elmo_base')  # 测试5


"2.3 get embedding of word that in sentence [it means the word have contexual information]"
# output embbedings of all words in sentence
sentencepath = '../test/data/text/sentence.txt'
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'w2v_base', outword='y')  # 测试1
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'glv_base', outword='y') 
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'fast_base', outword='y') 
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'rws_base', outword='y') 
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'cnt_base', outword='y') 
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', outword='y') # default model is gpt2_base  # 测试2
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'gpt2_medium', outword='y')  
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'gpt2_large', outword='y')   
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'gpt2_xl', outword='y')   
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'clip_base', outword='y')  # 测试3
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'bert_base', outword='y') # 模型适合做语境词向量  # 测试4
Vector = Text_vector(sentencepath, vectorType='word_in_sentence', label = 'elmo_base', outword='y') # 模型适合做语境词向量 # 测试5

# output embeddings of specific word in sentence
matchfile = '../test/data/text/matchfile.txt'
Vector = Text_vector(sentencepath, vectorType = 'word_in_sentence',matchfile=matchfile,outword='y') 

"2.4 get image embedding"
imagepath = '../test/data/image/imagefile.xlsx'
Vector = Image_vector(imagepath) # default model is clip_base
Vector = Image_vector(imagepath, outword = 'y') # default model is clip_base
Vector = Image_vector(imagepath, label='vgg_base') # default model is clip_base

"2.5 get chinese sentence embedding"
chinesepath = '../test/data/text/chinese.txt'
Vector = Text_vector(chinesepath, vectorType='sentence', label = 'gpt2_base',outword = 'y')
Vector = Text_vector(chinesepath, vectorType='sentence', label = 'elmo_base',outword = 'y')
