1. Download the pre-trained vector files from https://nlp.stanford.edu/projects/glove/. The detail information is as follows :
-------------------------------------------------------------------------------------------------------------------------------
This data is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/.
Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip
Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
-------------------------------------------------------------------------------------------------------------------------------
2. Unzip the vector files to this folder.
3. If you download the glove pre-trained vector from the official website, you should first convert the format to word2vec, then you can use the glove vector after format conversion to get embedding vectors
-------------------------------------------------------------------------------------------------------------------------------
glove_model = 'models/glove/glove.6B.50d.txt'
g2w_glove_model = 'models/glove/glove.6B.50d.g2w.txt'
glove2word2vec(glove_model, g2w_glove_model)
-------------------------------------------------------------------------------------------------------------------------------
