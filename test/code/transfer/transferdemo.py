# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:55:24 2023

@author: Dell
"""
from gensim.scripts.glove2word2vec import glove2word2vec
"""
transfer glove model into word2vec format
"""

"""
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors)
"""
glove_6B_50d = "models/glove/glove.6B.50d.txt"
glove_6B_100d = "models/glove/glove.6B.100d.txt"
glove_6B_200d = "models/glove/glove.6B.200d.txt"
glove_6B_300d = "models/glove/glove.6B.300d.txt"

glove_6B_50d_g2w = "models/glove/glove.6B.50d.g2w.txt"
glove_6B_100d_g2w = "models/glove/glove.6B.100d.g2w.txt"
glove_6B_200d_g2w = "models/glove/glove.6B.200d.g2w.txt"
glove_6B_300d_g2w = "models/glove/glove.6B.300d.g2w.txt"

glove2word2vec(glove_6B_50d, glove_6B_50d_g2w)
glove2word2vec(glove_6B_100d, glove_6B_100d_g2w)
glove2word2vec(glove_6B_200d, glove_6B_200d_g2w)
glove2word2vec(glove_6B_300d, glove_6B_300d_g2w)
"""
Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors)
"""
glove_42B_300d = "models/glove/glove.42B.300d.txt"
glove_42B_300d_g2w = "models/glove/glove.42B.300d.g2w.txt"
glove2word2vec(glove_42B_300d, glove_42B_300d_g2w)

"""
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)
"""
glove_840B_300d = "models/glove/glove.840B.300d.txt"
glove_840B_300d_g2w = "models/glove/glove.840B.300d.g2w.txt"
glove2word2vec(glove_840B_300d, glove_840B_300d_g2w)
"""
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors)
"""
glove_twitter_27B_25d = "models/glove/glove.twitter.27B.25d.txt"
glove_twitter_27B_50d = "models/glove/glove.twitter.27B.50d.txt"
glove_twitter_27B_100d = "models/glove/glove.twitter.27B.100d.txt"
glove_twitter_27B_200d = "models/glove/glove.twitter.27B.200d.txt"

glove_twitter_27B_25d_g2w = "models/glove/glove.twitter.27B.25d.g2w.txt"
glove_twitter_27B_50d_g2w = "models/glove/glove.twitter.27B.50d.g2w.txt"
glove_twitter_27B_100d_g2w = "models/glove/glove.twitter.27B.100d.g2w.txt"
glove_twitter_27B_200d_g2w = "models/glove/glove.twitter.27B.200d.g2w.txt"

glove2word2vec(glove_twitter_27B_25d, glove_twitter_27B_25d_g2w)
glove2word2vec(glove_twitter_27B_50d, glove_twitter_27B_50d_g2w)
glove2word2vec(glove_twitter_27B_100d, glove_twitter_27B_100d_g2w)
glove2word2vec(glove_twitter_27B_200d, glove_twitter_27B_200d_g2w)