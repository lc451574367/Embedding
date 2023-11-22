# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:28:02 2023
Dependency:
	tensorflow 1.14.0
	bert-serving-server
	bert-serving-client
	gensim
	scipy
	protobuf 3.20.1
	nltk
	numpy
	pandas
Module Help:
1. load model : 
	get_word2vec_model
	get_glove_model
	get_bert_model
    get_gpt2_model
2. Get different pre-trained model according different parameters :
	get_word2vec_model_by_diff_para
	get_glove_model_by_diff_para
	get_bert_model_by_diff_para
3. Get vector 
	get_word_vector_from_model
	Text_vector
	Clean_text

@author: Cheng Liu
"""
import gensim
import numpy as np
import os, signal
from model_path import *
from bert_serving.client import BertClient
import subprocess
from textprocess import *
import nltk
from nltk.corpus import stopwords
import re
from transformers import GPT2Model, GPT2Tokenizer
import torch


nltk.download("stopwords")
stopwords = stopwords.words('english')

"""
load models function
"""
def get_word2vec_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_path), binary=True)
    return model

def get_glove_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_path), binary=False)
    return model

def get_bert_model(model_name):
    print('---> model name : ' + model_name)
    command = 'bert-serving-start ' + '-model_dir '+ bert_path + '/' + model_name + '/' + ' -num_worker=1' + ' -max_seq_len=150'
    p = subprocess.Popen(command)
    print('< -- bert service start -->')
    bc = BertClient()
    print('< -- bert client start -->')
    return bc

def get_gpt2_model(model_name):
    print('---> model name : ' + model_name)
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model,tokenizer

def close_bert():
    # kill current bert-service
    # outinfo = subprocess.Popen('netstat -ano|findstr 5555', stdout=subprocess.PIPE, shell=True).communicate()
    # port = outinfo[0].decode().split()[-1]
    # p = subprocess.Popen('taskkill -pid {port}', shell=True)
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)


"""
Get different pre-trained model according different parameters
"""    
def get_word2vec_model_by_diff_para(corpus = 'GoogleNews'):
    size = 'none'
    dimension = 300
    model = get_word2vec_model(word2vec_GoogleNews)
    print('------------------------------------')
    print('Parameters :')
    print('---> corpus = ' + corpus + '\n---> dimension = ' + str(dimension) + '\n---> size = ' + size )
    print('------------------------------------\n')
    return model,size,dimension

def get_glove_model_by_diff_para(corpus = 'common', dimension = 300, size = 'L'):
    if corpus == 'common':
        dimension = 300
        if size == 'L':
            model = get_glove_model(glove_840B_300d_g2w)
        elif size == 'M':
            model = get_glove_model(glove_42B_300d_g2w)
        else:
            print('------------------------------------')
            print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
            print('---> Default size is [L], if you want to try other option, please input any one in {"L","M"}')
    elif corpus == 'wiki':
        size = 'none'
        if dimension == 300:
            model = get_glove_model(glove_6B_300d_g2w)
        elif dimension == 200:
            model = get_glove_model(glove_6B_200d_g2w)
        elif dimension == 100:
            model = get_glove_model(glove_6B_100d_g2w)
        elif dimension == 50:
            model = get_glove_model(glove_6B_50d_g2w)
        else :
            model = get_glove_model(glove_6B_300d_g2w)
            print('------------------------------------')
            print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
            print('---> Default dimension is 300, if you want to try other option, please input any one in {300,200,100,50}')
            dimension = 300
    elif corpus == 'twitter':
        size = 'none'
        if dimension == 200:
            model = get_glove_model(glove_twitter_27B_200d_g2w)
        elif dimension == 100:
            model = get_glove_model(glove_twitter_27B_100d_g2w)
        elif dimension == 50:
            model = get_glove_model(glove_twitter_27B_50d_g2w)
        elif dimension == 25:
            model = get_glove_model(glove_twitter_27B_25d_g2w)
        else :
            model = get_glove_model(glove_twitter_27B_200d_g2w)
            print('------------------------------------')
            print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
            print('---> Default dimension is 200, if you want to try other option, please input any one in {200,100,50,25}')
            dimension = 200
    else :
        dimension = 300
        size = 'L'
        model = get_glove_model(glove_840B_300d_g2w)
        print('------------------------------------')
        print('---> Warning: there is no corpus ' + corpus + ', turn to the default option.')
        print('---> Incorrect corpus! Default corpus is [common], if you want to try other option, please input the right corpus:{"common","wiki","twitter"}')
        corpus = 'common'
    print('------------------------------------')
    print('Parameters :')
    print('---> corpus = ' + corpus + '\n---> dimension = ' + str(dimension) + '\n---> size = ' + size )
    print('------------------------------------\n')
    return model,size,dimension


def get_bert_model_by_diff_para(layer = 12, dimension = 768, case = 'uncase', corpustype = 'none'):
    if case == 'uncase':
        if layer == 2:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_2_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_2_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_2_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_2_H_768d_uncase)
            else :
                model = get_bert_model(bert_L_2_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 4:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_4_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_4_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_4_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_4_H_768d_uncase)
            else :
                model = get_bert_model(bert_L_4_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 6:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_6_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_6_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_6_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_6_H_768d_uncase)   
            else :
                model = get_bert_model(bert_L_6_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 8:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_8_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_8_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_8_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_8_H_768d_uncase)  
            else :
                model = get_bert_model(bert_L_8_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 10:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_10_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_10_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_10_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_10_H_768d_uncase)   
            else :
                model = get_bert_model(bert_L_10_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 12:
            corpustype = 'none'
            if dimension == 128:
                model = get_bert_model(bert_L_12_H_128d_uncase)
            elif dimension == 256:
                model = get_bert_model(bert_L_12_H_256d_uncase)
            elif dimension == 512:
                model = get_bert_model(bert_L_12_H_512d_uncase)
            elif dimension == 768:
                model = get_bert_model(bert_L_12_H_768d_uncase)   
            else :
                model = get_bert_model(bert_L_12_H_768d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no dimension ' + str(dimension) + ', turn to the default option.')
                print('---> Default dimension is 768, if you want to try other option, please input any one in {128,256,512,768}')
                dimension = 768
        elif layer == 24:
            dimension = 1024
            if corpustype == 'none':
                model = get_bert_model(bert_L_24_H_1024d_uncase) 
            elif corpustype == 'wwm':
                model = get_bert_model(bert_L_24_H_1024d_uncase_wwm)
            else :
                model = get_bert_model(bert_L_24_H_1024d_uncase)
                print('------------------------------------')
                print('---> Warning: there is no corpustype ' + corpustype + ', turn to the default option.')
                print('---> Default corpustype is [none], if you want to try other option, please input any one in {"none","wwm"}')
                corpustype = 'none'
        else :
            print('------------------------------------')
            print('---> Warning: there is no layer ' + str(layer) + ', turn to the default option.')
            print('---> Default layer is 12, dimension is 768, if you want to try other option, please input any layer in {2,4,6,8,10,12,24}, dimension in {128,256,512,768,1024}')
            layer = 12
            model,layer,dimension,case,corpustype = get_bert_model_by_diff_para(layer = layer, dimension = dimension, case = case, corpustype = corpustype)
    elif case == 'case':
        if layer == 12:
            dimension = 768
            corpustype = 'none'
            model = get_bert_model(bert_L_12_H_768d_case) 
        elif layer == 24:
            dimension = 1024
            if corpustype == 'none':
                model = get_bert_model(bert_L_24_H_1024d_case) 
            elif corpustype == 'wwm':
                model = get_bert_model(bert_L_24_H_1024d_case_wwm)
            else :
                model = get_bert_model(bert_L_24_H_1024d_case) 
                print('------------------------------------')
                print('---> Warning: there is no corpustype ' + corpustype + ', turn to the default option.')
                print('---> Default corpustype is [none], if you want to try other option, please input any corpustype in {"none","wwm"}')
                corpustype = 'none'
        else :
            print('------------------------------------')
            print('---> Warning: there is no layer ' + str(layer) + ', turn to the default option.')
            print('---> Default layer is 12, dimension is 768, if you want to try other option, please input any layer in {12,24}')  
            layer == 12
            model,layer,dimension,case,corpustype = get_bert_model_by_diff_para(layer = layer, dimension = dimension, case = case, corpustype = corpustype)
    else :
        print('------------------------------------')
        print('---> Warning: there is no case ' + case + ', turn to the default option.')
        print('---> Default case is [uncase], if you want to try other option, please input any casetype in {"case","uncase"}')
        case = 'uncase'
        model,layer,dimension,case,corpustype = get_bert_model_by_diff_para(layer = layer, dimension = dimension, case = case, corpustype = corpustype)
    print('------------------------------------')
    print('Parameters :')
    print('---> layer = ' + str(layer) + '\n---> dimension = ' + str(dimension) + '\n---> case = ' + case + '\n---> corpustype = ' + corpustype)
    print('------------------------------------\n')
    return model,layer,dimension,case,corpustype


def get_word_vector_from_model(wordlist,model,dimension=300):
    """ 
        get vector of each word, if the word is not in this model, output nan 
	
    Parameters
    ----------
    wordlist : list
        list of words
    model : 
        glove, word2vec, bert model
    dimension : int
        
    Returns
    ----------
    wordvectorlist : list
        list of vecors of words
        
    """
    wordvectorlist = []
    for w in wordlist: 
        try :        
            vec = model[w]           
        except :
            print('------------------------------')
            print(w + ' is not in this model!')
            print('------------------------------')
            vec = np.empty(dimension)
            vec[:] = np.nan
                
        wordvectorlist.append(vec)
    return wordvectorlist


def Clean_text(sentencelist, ifstpw=0): 
    """
        clean the text by delete punctuation and stopwords
        
    Parameters
    ----------
    sentencelist : list
        list of text
    ifstpw : bool
        if delete words in stopwords {0,1} default is 0, do not delete words
    
    Returns
    ----------
    wordlist : nested list
        list of word list
    
    """
    # delete punctuation, keep only letters
    wordlist = [re.sub(r'[^\w\s]', ' ', sentence).split() for sentence in sentencelist]
    if ifstpw == 1:
    # delete words in stopwords
        for i in range(len(wordlist)):
            wordlist[i] = [w for w in wordlist[i] if w.lower() not in stopwords]
    elif ifstpw == 0:
        for i in range(len(wordlist)):
            wordlist[i] = [w for w in wordlist[i]]
    return wordlist

            
def Text_vector(file, modeltype = 'glove', corpus = 'common', layer = 12, dimension = 300, size = 'L', case = 'uncase', corpustype = 'none', filetype = 'csv', outword = 'n',ifstpw=0):            
    """
    	get vector of each row of text, if the word is not in this model, output nan 
     
    Parameters
    ----------
    file : file
        Text filepath
    modeltype : str, optional
        you can select different model {'bert','word2vec','glove'}, default option is 'glove' 
    corpus : str, optional, **only for glove and word2vec**
        you can select different corpus that used to train model:
        ---> glove model : {'common','wiki','twitter'}, default option is 'common'
        ---> word2vec model : {'GoogleNews'}
        ---> bert model do not have this option
    layer : int, optional, **only for bert**
        you can select different layer {2,4,6,8,10,12,24}, default optioni is 12
    dimension : int, optional
        you can select different dimension of pre-trained model:
        ---> bert model : {128, 256, 512, 768, 1024}, default option is 768
        ---> glove model : {25, 50, 100, 200, 300}, default option is 300
        ---> word2vec model : {300}
    size : str, optional, **only for glove**
        you can select different size of pre-trained model {'L', 'M'}
    case : str , optional, **only for bert**
        you can select different casetype {'uncase', 'case'}
    corpustype : str, optional, **only for bert model when layer = 24**
        you can select different corpus type {'none','wwm'}
    filetype : str, optional
        you can select outfile format {'txt','csv','xlsx'}, default format is csv
    outword : str, optional
        you can select if output words and vectors into one file {'n','y'}    
    ifstpw : bool
        if delete words in stopwords {0,1} default is 0, do not delete words

    Returns
    ----------
        output a file include Text(optional) and vectors
    
    Entire model selection
    ----------
    <bert model>
    'uncase'
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
    
    'case'
    27. modeltype:'bert' + layer:12 + dimenstion:768 + case:'case' + corpustype:'none' ---> [bert_L_12_H_768d_case] 
    28. modeltype:'bert' + layer:24 + dimenstion:1024 + case:'case' + corpustype:'none' ---> [bert_L_24_H_1024d_case] 
    29. modeltype:'bert' + layer:24 + dimenstion:1024 + case:'case' + corpustype:'wwm' ---> [bert_L_24_H_1024d_case_wwm] 
    
   <glove model>
   1. modeltype:'glove' + corpus:'common' + dimenstion:300 + size:'L' ---> [glove_840B_300d_g2w] [default]
   2. modeltype:'glove' + corpus:'common' + dimenstion:300 + size:'M' ---> [glove_42B_300d_g2w]
   3. modeltype:'glove' + corpus:'wiki' + dimenstion:300 + size:'none' ---> [glove_6B_300d_g2w]
   4. modeltype:'glove' + corpus:'wiki' + dimenstion:200 + size:'none' ---> [glove_6B_200d_g2w]
   5. modeltype:'glove' + corpus:'wiki' + dimenstion:100 + size:'none' ---> [glove_6B_100d_g2w]
   6. modeltype:'glove' + corpus:'wiki' + dimenstion:50 + size:'none' ---> [glove_6B_50d_g2w]
   7. modeltype:'glove' + corpus:'twitter' + dimenstion:200 + size:'none' ---> [glove_twitter_27B_200d_g2w]
   8. modeltype:'glove' + corpus:'twitter' + dimenstion:100 + size:'none' ---> [glove_twitter_27B_100d_g2w]
   9. modeltype:'glove' + corpus:'twitter' + dimenstion:50 + size:'none' ---> [glove_twitter_27B_50d_g2w]
   10. modeltype:'glove' + corpus:'twitter' + dimenstion:25 + size:'none' ---> [glove_twitter_27B_25d_g2w]
   
   <word2vec model>
   11. modeltype:'word2vec'+ corpus:'GoogleNews'+ dimenstion:300 + size:'none' ---> [word2vec_GoogleNews] 
    
    """
    sentencelist = readfiles(file)
    sentencevectorlist = []
    # select different model
    if modeltype == 'bert':
        model,layer,dimension,case,corpustype = get_bert_model_by_diff_para(layer=layer,dimension=dimension,case=case,corpustype=corpustype) 
        print('< -- get model success! -- >')
        # get sentence vector
        sentencevectorarr = model.encode(sentencelist)
        sentencevectorlist = sentencevectorarr.tolist()
        # model name
        modelname = modeltype + '_L' + str(layer) + '_' + str(dimension) + '_' + case + '_' + corpustype
    
    elif modeltype == 'gpt2':
        model,tokenizer = get_gpt2_model(gpt2_path)
        print('< -- get model success! -- >')
        # divide text
        chunk_size = 1000
        chunks = [sentencelist[i:i + chunk_size] for i in range(0, len(sentencelist), chunk_size)]
        # process by chunks
        all_outputs = []
        for chunk in chunks:
            sentence_ids = tokenizer.encode(chunk, return_tensors="pt")
            with torch.no_grad():
                output = model(sentence_ids)
                all_outputs.append(output)
        # merge outputs
        for out in all_outputs:
            sentencevectorlist.extend(out.last_hidden_state.numpy().squeeze())
        modelname = modeltype
    
    elif modeltype == 'glove' or modeltype == 'word2vec':
        if modeltype == 'glove':
            model,size,dimension = get_glove_model_by_diff_para(corpus = corpus, dimension = dimension, size = size)
            print('< -- get model success! -- >')
        elif modeltype == 'word2vec':
            model,size,dimension = get_word2vec_model_by_diff_para(corpus = corpus)
            print('< -- get model success! -- >')
        
        # clean sentence by delete punctation and stopwords
        wordlist = Clean_text(sentencelist,ifstpw)
        
        for wl in wordlist:
            wlvec = get_word_vector_from_model(wordlist = wl,model = model,dimension=dimension)
            wlvec = [v for v in wlvec if np.isnan(v).any() == 0]
            if len(wlvec) == 0:
                vec = np.empty(dimension)
                vec[:] = np.nan
                sentencevectorlist.append(vec)
            else :
                sentencevectorlist.append(np.mean(wlvec,axis=0))
        modelname = modeltype + '_' + corpus + '_' + str(dimension) + '_' + size
        
    else :
        print('please input the right modeltype:{"bert","glove","word2vec"}')
        
    if not os.path.exists('results'):
        os.makedirs('results')
    outpath = 'results/Vector_' + modelname + '.' + filetype
    savefiles(sentencelist,sentencevectorlist,outpath,filetype,outword)
    print('< -- results has output -- >')
    
    # kill current bert-service
    # print('< -- bert service close -- >')
