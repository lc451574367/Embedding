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

<bert model>

1. modeltype:'bert' + layer:2 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_128d_uncase] 
2. modeltype:'bert' + layer:2 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_256d_uncase] 
3. modeltype:'bert' + layer:2 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_512d_uncase] 
4. modeltype:'bert' + layer:2 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_768d_uncase] 
5. modeltype:'bert' + layer:4 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_128d_uncase] 
6. modeltype:'bert' + layer:4 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_256d_uncase] 
7. modeltype:'bert' + layer:4 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_512d_uncase] 
8. modeltype:'bert' + layer:4 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_768d_uncase] 
9. modeltype:'bert' + layer:6 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_128d_uncase] 
10. modeltype:'bert' + layer:6 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_256d_uncase] 
11. modeltype:'bert' + layer:6 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_512d_uncase] 
12. modeltype:'bert' + layer:6 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_768d_uncase] 
13. modeltype:'bert' + layer:8 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_128d_uncase] 
14. modeltype:'bert' + layer:8 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_256d_uncase] 
15. modeltype:'bert' + layer:8 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_512d_uncase] 
16. modeltype:'bert' + layer:8 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_768d_uncase] 
17. modeltype:'bert' + layer:10 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_128d_uncase] 
18. modeltype:'bert' + layer:10 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_256d_uncase] 
19. modeltype:'bert' + layer:10 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_512d_uncase] 
20. modeltype:'bert' + layer:10 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_768d_uncase] 
21. modeltype:'bert' + layer:12 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_128d_uncase] 
22. modeltype:'bert' + layer:12 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_256d_uncase] 
23. modeltype:'bert' + layer:12 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_512d_uncase] 
24. modeltype:'bert' + layer:12 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_768d_uncase] 
25. modeltype:'bert' + layer:24 + dimenstion:1024 + case:'uncase' + corpustype:'none' ---> [bert_L_24_H_1024d_uncase] 
26. modeltype:'bert' + layer:24 + dimenstion:1024 + case:'uncase' + corpustype:'wwm' ---> [bert_L_24_H_1024d_uncase_wwm] 

"""


"-------------------------------------4. gpt2 models-------------------------------------------"
gpt2_online_name = "gpt2"
gpt2_local_path = "models/gpt2"
"-------------------------------------5. clip models-------------------------------------------"

clip_online_name = "openai/clip-vit-base-patch32"
clip_local_path = "models/clip/clip-vit-base-patch32"
