# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:36:03 2023
Dependency:
Module Help:
    
@author: Cheng Liu

"""
import os
models = 'models/'

"-------------------------------------1. word2vec models-------------------------------------------"
"""
pre-trained vectors trained on part of Google News dataset (about 100 billion words). 
The model contains 300-dimensional vectors for 3 million words and phrases
"""
word2vec_GoogleNews = os.path.join(models,'word2vec/GoogleNews-vectors-negative300.bin')

"-------------------------------------2. glove models-------------------------------------------"

"""
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors)
"""
glove_6B_50d_g2w = os.path.join(models,'glove/glove.6B.50d.g2w.txt')
glove_6B_100d_g2w = os.path.join(models,'glove/glove.6B.100d.g2w.txt')
glove_6B_200d_g2w = os.path.join(models,'glove/glove.6B.200d.g2w.txt')
glove_6B_300d_g2w = os.path.join(models,'glove/glove.6B.300d.g2w.txt')
"""
Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors)
"""
glove_42B_300d_g2w = os.path.join(models,'glove/glove.42B.300d.g2w.txt')
"""
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)
"""
glove_840B_300d_g2w = os.path.join(models,'glove/glove.840B.300d.g2w.txt')
"""
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors)
"""
glove_twitter_27B_25d_g2w = os.path.join(models,'glove/glove.twitter.27B.25d.g2w.txt')
glove_twitter_27B_50d_g2w = os.path.join(models,'glove/glove.twitter.27B.50d.g2w.txt')
glove_twitter_27B_100d_g2w = os.path.join(models,'glove/glove.twitter.27B.100d.g2w.txt')
glove_twitter_27B_200d_g2w = os.path.join(models,'glove/glove.twitter.27B.200d.g2w.txt')

"-------------------------------------3. bert models-------------------------------------------"
"""
bert_base_uncase totally 24
"""
bert_path = 'models/bert'

bert_L_2_H_128d_uncase = "uncased_L-2_H-128_A-2"
bert_L_2_H_256d_uncase = "uncased_L-2_H-256_A-4"
bert_L_2_H_512d_uncase = "uncased_L-2_H-512_A-8"
bert_L_2_H_768d_uncase = "uncased_L-2_H-768_A-12"

bert_L_4_H_128d_uncase = "uncased_L-4_H-128_A-2"
bert_L_4_H_256d_uncase = "uncased_L-4_H-256_A-4"
bert_L_4_H_512d_uncase = "uncased_L-4_H-512_A-8"
bert_L_4_H_768d_uncase = "uncased_L-4_H-768_A-12"

bert_L_6_H_128d_uncase = "uncased_L-6_H-128_A-2"
bert_L_6_H_256d_uncase = "uncased_L-6_H-256_A-4"
bert_L_6_H_512d_uncase = "uncased_L-6_H-512_A-8"
bert_L_6_H_768d_uncase = "uncased_L-6_H-768_A-12"

bert_L_8_H_128d_uncase = "uncased_L-8_H-128_A-2"
bert_L_8_H_256d_uncase = "uncased_L-8_H-256_A-4"
bert_L_8_H_512d_uncase = "uncased_L-8_H-512_A-8"
bert_L_8_H_768d_uncase = "uncased_L-8_H-768_A-12"

bert_L_10_H_128d_uncase = "uncased_L-10_H-128_A-2"
bert_L_10_H_256d_uncase = "uncased_L-10_H-256_A-4"
bert_L_10_H_512d_uncase = "uncased_L-10_H-512_A-8"
bert_L_10_H_768d_uncase = "uncased_L-10_H-768_A-12"

bert_L_12_H_128d_uncase = "uncased_L-12_H-128_A-2"
bert_L_12_H_256d_uncase = "uncased_L-12_H-256_A-4"
bert_L_12_H_512d_uncase = "uncased_L-12_H-512_A-8"
bert_L_12_H_768d_uncase = "uncased_L-12_H-768_A-12"

"""
bert_base_case 
"""
bert_L_12_H_768d_case = "cased_L-12_H-768_A-12"

"""
bert_large 
"""
bert_L_24_H_1024d_uncase = "uncased_L-24_H-1024_A-16"
bert_L_24_H_1024d_uncase_wwm = "wwm_uncased_L-24_H-1024_A-16"

bert_L_24_H_1024d_case = "cased_L-24_H-1024_A-16"
bert_L_24_H_1024d_case_wwm = "wwm_cased_L-24_H-1024_A-16"

"""
bert_multilingual 
"""
bert_L_12H_768d_case_multi = "multilingual_L-12_H-768_A-12"

"-------------------------------------3. bert models-------------------------------------------"
gpt2_path = 'models/gpt2'
