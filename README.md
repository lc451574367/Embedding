# Embedding

This repository includes a set of python scripts for getting embedding vectors of word and sentence by word2vec, glove and bert. The target audience is NLP and calculation of semantic distance between words, word pairs, and texts.

## Dependencies

To run the python scripts : 
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim numpy scipy pandas transformers torch openpyxl nltk Pillow allennlp string re torchvision tabulate
```

## Installation

After downloading and installing dependencies, download this repository using the following command line:

```
$ git clone https://github.com/lc451574367/Embedding
```

## How to run the codes

The python script `scripts_allstep.py` contains all functions.

## Note
The bert service is often cause misuse of BertClient in multi-thread/process environment. You can’t reuse one BertClient among multiple threads/processes. If you want to use two different bert models in a row, you must close the current model after using it and then use the next one. 
