# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:28:02 2023
Dependency:
Module Help:


@author: Cheng Liu
"""
import gensim
import numpy as np
import os, signal
from model_path import *
from bert_serving.client import BertClient
import subprocess
from textprocess import *
from nltk.corpus import stopwords
import re

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
    command = 'bert-serving-start ' + '-model_dir '+ bert_path + '/' + model_name + '/' + ' -num_worker=1'
    p = subprocess.Popen(command)
    print('< -- bert service start -->')
    bc = BertClient()
    print('< -- bert client start -->')
    return bc

def close_bert():
    # kill current bert-service
    outinfo = subprocess.Popen('netstat -ano|findstr 5555', stdout=subprocess.PIPE, shell=True).communicate()
    port = outinfo[0].decode().split()[-1]
    p = subprocess.Popen('taskkill -pid {port}', shell=True)


"""
load different pre-trained model according different parameters
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
    # get vector of each word, if the word is not in this model, will output nan  
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

"""
get word vector by different pre-trained model
"""         
def Word_vector(file, modeltype = 'glove', corpus = 'common', dimension = 300, size = 'L', filetype = 'csv'):
    """
    
    Parameters
    ----------
    file : file
        wordlist filepath
    modeltype : str, optional
        you can select different algorithm {'word2vec','glove'}, default option is 'glove' 
    corpus : str, optional
        you can select different corpus that used to train model {'common','wiki','twitter'}, default option is 'common'
    dimension : int, optional
        you can select different dimension of pre-trained model {25, 50, 100, 200, 300}, default option is 300
    size : str, optional
        you can select different size of pre-trained model {'L', 'M'}
        ** only glove model **
    filetype : str, optional
        you can select outfile format {'txt','csv','xlsx'}, defult format is txt
    
    Returns
    ----------
    outpath
        output a file include word and vector
        
    Entire model selection
    ----------
    <glove model>
    1. modeltype:'glove'+ corpus:'common'+ dimenstion:300 + size:'L' ---> [glove_840B_300d_g2w] [default]
    2. modeltype:'glove'+ corpus:'common'+ dimenstion:300 + size:'M' ---> [glove_42B_300d_g2w]
    3. modeltype:'glove'+ corpus:'wiki'+ dimenstion:300 + size:'none' ---> [glove_6B_300d_g2w]
    4. modeltype:'glove'+ corpus:'wiki'+ dimenstion:200 + size:'none' ---> [glove_6B_200d_g2w]
    5. modeltype:'glove'+ corpus:'wiki'+ dimenstion:100 + size:'none' ---> [glove_6B_100d_g2w]
    6. modeltype:'glove'+ corpus:'wiki'+ dimenstion:50 + size:'none' ---> [glove_6B_50d_g2w]
    7. modeltype:'glove'+ corpus:'twitter'+ dimenstion:200 + size:'none' ---> [glove_twitter_27B_200d_g2w]
    8. modeltype:'glove'+ corpus:'twitter'+ dimenstion:100 + size:'none' ---> [glove_twitter_27B_100d_g2w]
    9. modeltype:'glove'+ corpus:'twitter'+ dimenstion:50 + size:'none' ---> [glove_twitter_27B_50d_g2w]
    10. modeltype:'glove'+ corpus:'twitter'+ dimenstion:25 + size:'none' ---> [glove_twitter_27B_25d_g2w]
    
    <word2vec model>
    11. modeltype:'word2vec'+ corpus:'GoogleNews'+ dimenstion:300 + size:'none' ---> [word2vec_GoogleNews] 
    """
    
    wordlist = readfiles(file)
    wordvectorlist = []
    
    # select different model
    if modeltype == 'glove':
        model,size,dimension = get_glove_model_by_diff_para(corpus = corpus, dimension = dimension, size = size)
    elif modeltype == 'word2vec':
        model,size,dimension = get_word2vec_model_by_diff_para(corpus = corpus)
    else :
        print('please input the right modeltype:{"glove","word2vec"}')
    
    wordvectorlist = get_word_vector_from_model(wordlist = wordlist,model = model,dimension=dimension)
    
    # output to file        
    modelname = modeltype + '_' + corpus + '_' + str(dimension) + '_' + size
    if not os.path.exists('results'):
        os.makedirs('results')
    outpath = 'results/wordVector_' + modelname + '.' + filetype
    savefiles(wordlist,wordvectorlist,outpath,filetype)        

def Clean_sentence(sentencelist): 
    # delete punctuation
    wordlist = [re.sub(r'[^\w\s]', ' ', sentence).split() for sentence in sentencelist]
    # delete stopwords
    for i in range(len(wordlist)):
        wordlist[i] = [w for w in wordlist[i] if w.lower() not in stopwords]
    return wordlist
            
def Sentence_vector(file, modeltype = 'bert', corpus = 'common', layer = 12, dimension = 768, size = 'L', case = 'uncase', corpustype = 'none', filetype = 'csv'):            
    
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
    
    elif modeltype == 'glove' or modeltype == 'word2vec':
        if modeltype == 'glove':
            model,size,dimension = get_glove_model_by_diff_para(corpus = corpus, dimension = dimension, size = size)
        elif modeltype == 'word2vec':
            model,size,dimension = get_word2vec_model_by_diff_para(corpus = corpus)
        
        # clean sentence by delete punctation and stopwords
        wordlist = Clean_sentence(sentencelist)
        
        for wl in wordlist:
            wlvec = get_word_vector_from_model(wordlist = wl,model = model,dimension=dimension)
            wlvec = [v for v in wlvec if np.isnan(v).any() == 0]
            sentencevectorlist.append(np.mean(wlvec,axis=0))
        modelname = modeltype + '_' + corpus + '_' + str(dimension) + '_' + size
        
    else :
        print('please input the right modeltype:{"bert","glove","word2vec"}')
        
    if not os.path.exists('results'):
        os.makedirs('results')
    outpath = 'results/sentenceVector_' + modelname + '.' + filetype
    savefiles(sentencelist,sentencevectorlist,outpath,filetype)
    print('< -- results has output -- >')
    
    # kill current bert-service
    close_bert()
    print('< -- bert service close -- >')
    # pid = os.getpid()
    # os.kill(pid, signal.SIGTERM)
    

    
    
    
    
    
    
            