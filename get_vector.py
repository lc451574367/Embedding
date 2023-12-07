# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:28:02 2023
Dependency:
	gensim
	scipy
	nltk
	numpy
	pandas
    transformers
    torch
    openpyxl
    Pillow
Module Help:
1. load model : 
	get_word2vec_model
	get_glove_model
	get_bert_model
    get_gpt2_model
    get_clip_model
2. Get different pre-trained model according different parameters :
	get_word2vec_model_by_diff_para
	get_glove_model_by_diff_para
	get_bert_model_by_diff_para
3. Get vector 
	get_word_vector_from_model
	Text_vector
	Clean_text
    Image_vector

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
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer, CLIPModel, AutoProcessor,AutoTokenizer
import torch
from PIL import Image


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
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return model,tokenizer

def get_gpt2_model():
    try :
        model = GPT2Model.from_pretrained(gpt2_online_name)
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_online_name)
    except :
        print('\n------------------------------------')
        print("Can't visit models online, load the models from local")
        print('------------------------------------')

        model = GPT2Model.from_pretrained(gpt2_local_path)
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_local_path)

    return model,tokenizer

def get_clip_model():
    try:
        model = CLIPModel.from_pretrained(clip_online_name)
        processor = AutoProcessor.from_pretrained(clip_online_name)
        tokenizer = AutoTokenizer.from_pretrained(clip_online_name)

    except : 
        print('\n------------------------------------')
        print("Can't visit models online, load the models from local")
        print('------------------------------------')
        model = CLIPModel.from_pretrained(clip_local_path)
        processor = AutoProcessor.from_pretrained(clip_local_path)
        tokenizer = AutoTokenizer.from_pretrained(clip_local_path)

    return model,processor,tokenizer

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
    """
    <word2vec model>
    modeltype:'word2vec'+ corpus:'GoogleNews'+ dimenstion:300 + size:'none' ---> [word2vec_GoogleNews] 
    """
    return model,size,dimension

def get_glove_model_by_diff_para(corpus = 'common', dimension = 300, size = 'L'):
    
    path = 'models/glove/'
    if corpus == 'twitter' :
        byte = '27B'
        cname = 'twitter'
    
    else:
        cname = ''
        if corpus == 'wiki' :
            byte = '6B'
       
        elif corpus == 'common' : 
            if size == 'L':
                byte = '840B'
            elif size == 'M':
                byte = '42B'
    
        
    name = 'glove.' + cname + byte + '.' + str(dimension) + 'd.g2w.txt'
    local_path = path + name
    
    try : 
        model = get_glove_model(local_path)
        print('------------------------------------')
        print('Parameters :')
        print('---> corpus = ' + corpus + '\n---> dimension = ' + str(dimension) + '\n---> size = ' + size )
        print('------------------------------------\n')

    except :
        print('!!the parameter you input are error!!')
        print('entire model information:')
        print("1. modeltype:'glove' + corpus:'common' + dimenstion:300 + size:'L' ---> [glove_840B_300d_g2w] [default]")
        print("2. modeltype:'glove' + corpus:'common' + dimenstion:300 + size:'M' ---> [glove_42B_300d_g2w]")
        print("3. modeltype:'glove' + corpus:'wiki' + dimenstion:300 + size:'none' ---> [glove_6B_300d_g2w]")
        print("4. modeltype:'glove' + corpus:'wiki' + dimenstion:200 + size:'none' ---> [glove_6B_200d_g2w]")
        print("5. modeltype:'glove' + corpus:'wiki' + dimenstion:100 + size:'none' ---> [glove_6B_100d_g2w]")
        print("6. modeltype:'glove' + corpus:'wiki' + dimenstion:50 + size:'none' ---> [glove_6B_50d_g2w]")
        print("7. modeltype:'glove' + corpus:'twitter' + dimenstion:200 + size:'none' ---> [glove_twitter_27B_200d_g2w]")
        print("8. modeltype:'glove' + corpus:'twitter' + dimenstion:100 + size:'none' ---> [glove_twitter_27B_100d_g2w]")
        print("9. modeltype:'glove' + corpus:'twitter' + dimenstion:50 + size:'none' ---> [glove_twitter_27B_50d_g2w]")
        print("10. modeltype:'glove' + corpus:'twitter' + dimenstion:25 + size:'none' ---> [glove_twitter_27B_25d_g2w]")
        print('------------------------------------')
        dimension = 300
        size = 'L'
        corpus = 'common'
        model = get_glove_model('models/glove/glove.840B.300d.g2w.txt')
    return model,size,dimension
    

def get_bert_model_by_diff_para(layer = 12, dimension = 768, case = 'uncase', corpustype = 'none',language = 'en'):
    path = 'models/bert/'
    
    if language == 'en':
    
        if dimension == 128 :
            heads = 2
        elif dimension == 256 : 
            heads = 4
        elif dimension == 512:
            heads = 8
        elif dimension == 768:
            heads = 12
        else :
            print("default dimension is 768")
            dimension = 768
            heads = 12
        
        name = 'bert_' + case + 'd_L-' + str(layer) + '_H-' + str(dimension) + '_A-' + str(heads)
        online_name = 'google/bert_' + name
        local_path = path + name
    elif language == 'cn':
        dimension = 768
        case = ''
        name = 'bert-base-chinese'
        online_name = 'bert-base-chinese'
        local_path = path + name
        
    
    try:
        tokenizer,model = get_bert_model(online_name)
    except:
        print('\n------------------------------------')
        print("can't visit models online.")
        print('------------------------------------\n')
        try : 
            print('------------------------------------')
            print("> load the model from local path.")
            tokenizer,model = get_bert_model(local_path)
            print("< -- load local model success! -- >")
        except :
            print("We don't provide this model in local, please download the model from https://huggingface.co/")
            print("The models we provide in local as follows:")
            print("1. modeltype:'bert' + layer:12 + dimenstion:768 + case:'uncase' + corpustype:'none' + language:'en' ---> models/bert/bert_uncased_L-12_H-768_A-12")
            print("2. modeltype:'bert' + layer:12 + dimenstion:768 + case:'' + corpustype:'' + language:'cn' ---> models/bert/bert-base-chinese")
            print('------------------------------------')
            print("You can download models from https://huggingface.co/ that have parameter as follows:")
            print("1. modeltype:'bert' + layer:2 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_128d_uncase]")
            print("2. modeltype:'bert' + layer:2 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_256d_uncase]")
            print("3. modeltype:'bert' + layer:2 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_512d_uncase]")
            print("4. modeltype:'bert' + layer:2 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_2_H_768d_uncase]")
            print("5. modeltype:'bert' + layer:4 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_128d_uncase]")
            print("6. modeltype:'bert' + layer:4 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_256d_uncase]")
            print("7. modeltype:'bert' + layer:4 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_512d_uncase]")
            print("8. modeltype:'bert' + layer:4 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_4_H_768d_uncase]")
            print("9. modeltype:'bert' + layer:6 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_128d_uncase]")
            print("10. modeltype:'bert' + layer:6 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_256d_uncase]")
            print("11. modeltype:'bert' + layer:6 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_512d_uncase]")
            print("12. modeltype:'bert' + layer:6 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_6_H_768d_uncase]")
            print("13. modeltype:'bert' + layer:8 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_128d_uncase]")
            print("14. modeltype:'bert' + layer:8 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_256d_uncase]")
            print("15. modeltype:'bert' + layer:8 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_512d_uncase]")
            print("16. modeltype:'bert' + layer:8 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_8_H_768d_uncase]")
            print("17. modeltype:'bert' + layer:10 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_128d_uncase]")
            print("18. modeltype:'bert' + layer:10 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_256d_uncase]")
            print("19. modeltype:'bert' + layer:10 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_512d_uncase]")
            print("20. modeltype:'bert' + layer:10 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_10_H_768d_uncase]")
            print("21. modeltype:'bert' + layer:12 + dimenstion:128 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_128d_uncase]")
            print("22. modeltype:'bert' + layer:12 + dimenstion:256 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_256d_uncase]")
            print("23. modeltype:'bert' + layer:12 + dimenstion:512 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_512d_uncase]")
            print("24. modeltype:'bert' + layer:12 + dimenstion:768 + case:'uncase' + corpustype:'none' ---> [bert_L_12_H_768d_uncase]")

    
    print('------------------------------------')
    print('Parameters :')
    print('---> layer = ' + str(layer) + '\n---> dimension = ' + str(dimension) + '\n---> case = ' + case + '\n---> corpustype = ' + corpustype)
    print('------------------------------------\n')
    return tokenizer,model,layer,dimension,case,corpustype

   

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

            
def Text_vector(file, modeltype = 'glove', corpus = 'common', layer = 12, dimension = 300, size = 'L', case = 'uncase', corpustype = 'none', filetype = 'csv', outword = 'n',ifstpw=0,language = 'en'):            
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
        
    """
    sentencelist = readfiles(file)
    sentencevectorlist = []
    
    # select different model
    if modeltype == 'bert':
        model,tokenizer,layer,dimension,case,corpustype = get_bert_model_by_diff_para(layer=layer,dimension=dimension,case=case,corpustype=corpustype,language=language) 
        print('< -- get model success! -- >')
        # get sentence vector
        outputslist = []
        for s in sentencelist:
            input_ids = tokenizer.encode(s,add_special_tokens=True,return_tensors='pt')
            outputs = model(input_ids).last_hidden_state[:,-2,:].detach().numpy()
            outputslist.append(outputs)
        # model name
        sentencevectorlist = [arr.flatten() for arr in outputslist]
        modelname = modeltype + '_L' + str(layer) + '_' + str(dimension) + '_' + case + '_' + corpustype + '_' + language
    
    elif modeltype == 'gpt2':
        model,tokenizer = get_gpt2_model()
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
        outputslist = []
        for out in all_outputs:
            outputslist.extend(out.last_hidden_state.numpy().squeeze())
        sentencevectorlist = [arr.flatten() for arr in outputslist]
        modelname = modeltype
    
    elif modeltype == 'clip':
        model,_,tokenizer = get_clip_model()
        
        inputs = tokenizer(sentencelist, padding=True, return_tensors="pt")
        sentencevectorlist = model.get_text_features(**inputs).detach().numpy()
        dimension = 512
        modelname = modeltype + '_' + str(dimension)
        
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
    print(outpath)

    
def Image_vector(imagefile, filetype = 'csv', outword = 'n'):   
    """
    	get vector of each row of image
        
    Parameters
    ----------
    imagefile : file
        image path
    filetype : str, optional
        you can select outfile format {'txt','csv','xlsx'}, default format is csv
    outword : str, optional
        you can select if output words and vectors into one file {'n','y'}    
    
    Returns
    ----------
        output a file include Text(optional) and vectors
    """ 

    imagelist = readfiles(imagefile)
    image = [Image.open(f) for f in imagelist]
    
    model,processor,_ = get_clip_model()
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**inputs).detach().numpy()

    if not os.path.exists('results'):
        os.makedirs('results')
    outpath = 'results/Vector_clip_512.' + filetype
    
    savefiles(imagelist,image_features,outpath,filetype,outword)
    print('< -- results has output -- >')
