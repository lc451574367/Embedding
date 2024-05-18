# -*- coding: utf-8 -*-
"""
Module Help: 
	get_word_vector_from_model
	Text_vector
	Clean_text
    Image_vector

@author: Cheng Liu
"""
import sys,os
sys.path.append(os.getcwd())
from feature.load_model import *

import re
import torchvision.transforms as transforms

import nltk
from nltk.corpus import stopwords

# 尝试获取停用词，如果失败则下载后重试
try:
    stop_words = stopwords.words('english')
except LookupError:  # NLTK 在未找到资源时会抛出 LookupError
    print("Stopwords resource not found. Downloading now...")
    nltk.download('stopwords')
    stop_words = stopwords.words('english')  # 重新尝试加载停用词

def get_word_vector_from_model(wordlist,model,dimension=300):
    """ 
        get vector of each word, if the word is not in this model, output nan 
	
    Parameters
    ----------
    wordlist : list
        list of words
    model : 
        glove, word2vec, fastText
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


def get_vector_from_bert_model(modelpath,vectorType,chunks):
    """
        get vector from bert model by different vectorType
        
    Parameters
    ----------
    modelpath : str
        bert model path
    vectorType : str
        select your vector type of data
            'word': get vector of each single word, data format is one word each row
            'sentence': get vector of each sentence, data format is some words each row
            'word_in_sentence': get vector of each word that in sentence (it means the word has a contextual information), data format is some words each row
    chunks : str in list
        divide all data into different chunk
    
    Returns
    ----------
    vectorlist : 
        list of vector
    
    """
    model,tokenizer = get_bert_model(modelpath) 
    print('< -- get bert model success! -- >')
    all_outputs = []
    
    if vectorType == 'sentence':
    #process by chunks
        outputs = []
        for chunk in chunks:
            input_ids = tokenizer.encode(chunk,add_special_tokens=True,return_tensors='pt')
            outputs = model(input_ids).last_hidden_state[0,1:-1,:].detach().numpy()
            all_outputs.append(outputs)
    
    elif vectorType == 'word' or vectorType == 'word_in_sentence':
    #process by chunks
        for chunk in chunks:
            if vectorType == 'word':
                chunk = [c.split() for c in chunk]
            input_ids = [tokenizer.encode(c,add_special_tokens=True,return_tensors='pt') for c in chunk]
            outputs = [model(inp).last_hidden_state[0,1:-1,:].detach().numpy() for inp in input_ids]
            all_outputs.append(outputs)
    
    return all_outputs

def get_vector_from_gpt2_model(modelpath,vectorType,chunks):
    model,tokenizer = get_gpt2_model(modelpath)
    print('< -- get gpt2 model success! -- >')
    all_outputs = []
        
    if vectorType == 'sentence':
    #process by chunks
        for chunk in chunks:
            input_ids = tokenizer.encode(chunk,add_special_tokens=True,return_tensors='pt')
            outputs = model(input_ids).last_hidden_state[0,:,:].detach().numpy()
            all_outputs.append(outputs)
            
    elif vectorType == 'word' or vectorType == 'word_in_sentence':
    #process by chunks
        for chunk in chunks:
            chunk = [c.split() for c in chunk]
            input_ids = [tokenizer.encode(c,add_special_tokens=True,return_tensors='pt') for c in chunk]
            outputs = [model(inp).last_hidden_state[0,:,:].detach().numpy() for inp in input_ids]
            all_outputs.append(outputs)
        
    return all_outputs

def get_vector_from_clip_model(modelpath,vectorType,chunks,multi='image'):
    if multi == 'text':
        model,_,tokenizer = get_clip_model(modelpath)
        print('< -- get clip model success! -- >')
        all_outputs = []
        
        if vectorType == 'sentence':
        #process by chunks
            for chunk in chunks:
                input_ids = tokenizer(chunk, padding=True, return_tensors="pt")
                outputs = model.get_text_features(**input_ids).detach().numpy()
                all_outputs.append(outputs)

        elif vectorType == 'word' or vectorType == 'word_in_sentence':
            for chunk in chunks:
                chunk = [c.split() for c in chunk]
                input_ids = [tokenizer(c, padding=True, return_tensors="pt") for c in chunk]
                outputs = [model.get_text_features(**inp).detach().numpy() for inp in input_ids]
                all_outputs.append(outputs)
                        
    elif multi == 'image':
        model,processor,_ = get_clip_model(modelpath)
        print('< -- get clip model success! -- >')
        all_outputs = []
        
        for chunk in chunks:
            inputs_ids = processor(images=chunk, return_tensors="pt", padding=True)
            outputs = model.get_image_features(**inputs_ids).detach().numpy()
            all_outputs.append(outputs)
        
    return all_outputs

def get_vector_from_pretrained_model(modeltype,modelpath,vectorType,chunks,dimension):
    if modeltype == 'glv':
        model = get_glove_model(modelpath)
        print('< -- get glove model success! -- >')
    elif modeltype == 'w2v':
        model = get_gensim_model(modelpath,binary = True)
        print('< -- get word2vec model success! -- >')
    elif modeltype == 'fast':
        model = get_gensim_model(modelpath,binary = False)
        print('< -- get fastText model success! -- >')
    elif modeltype == 'cnt':
        model = get_gensim_model(modelpath,binary = False)
        print('< -- get conceptNet model success! -- >')
    elif modeltype == 'rws':
        model = get_gensim_model(modelpath,binary = False)
        print('< -- get conceptNet model success! -- >')
        
    all_outputs = []
    
    if vectorType == 'word' or vectorType == 'word_in_sentence':
        for chunk in chunks:
            chunk = [c.split() for c in chunk]
            outputs = []
            for wl in chunk:
                wlvec = get_word_vector_from_model(wordlist = wl,model = model,dimension=dimension)
                outputs.append(np.array(wlvec))
            all_outputs.append(outputs)
    
    elif vectorType == 'sentence':
        for chunk in chunks:
            chunk = [c.split() for c in chunk]
            outputs = []
            for wl in chunk:
                wlvec = get_word_vector_from_model(wordlist = wl,model = model,dimension=dimension)
                wlvec = [v for v in wlvec if np.isnan(v).any() == 0]
                if len(wlvec) == 0:
                    vec = np.empty(dimension)
                    vec[:] = np.nan
                    outputs.append(vec)
                else :
                    outputs.append(np.mean(wlvec,axis=0))
                
            all_outputs.append(outputs)
    
    return all_outputs    

def get_vector_from_elmo_model(modelpath,vectorType,chunks):
    model = get_elmo_model(modelpath)
    print('< -- get elmo model success! -- >')
    
    all_outputs = []
    
    if vectorType == 'word':
        # process by chunks
        for chunk in chunks:
            chunk = [c.split() for c in chunk]
            chunk = [[c] for c in chunk]
            character_ids = [batch_to_ids(c) for c in chunk]
            outputs = [model(cha)['elmo_representations'][0].detach().numpy().squeeze() for cha in character_ids]
            
            all_outputs.append(outputs)
                
    elif vectorType == 'sentence' :
        for chunk in chunks:
            chunk = [[c] for c in chunk]
            character_ids = batch_to_ids(chunk)
            out = model(character_ids)['elmo_representations'][0].detach().numpy()
            outputs = []
            for o in out:
                non_zero_rows = [i for i in range(o.shape[0]) if np.any(o[i])]
                outputs.append(o[non_zero_rows].squeeze())
            
            outputs = np.array(outputs)
            all_outputs.append(outputs)
    
    elif vectorType == 'word_in_sentence':
        for chunk in chunks:
            chunk = [c.split() for c in chunk]
            character_ids = batch_to_ids(chunk)
            out = model(character_ids)['elmo_representations'][0].detach().numpy()
            outputs = []
            for o in out:
                non_zero_rows = [i for i in range(o.shape[0]) if np.any(o[i])]
                outputs.append(o[non_zero_rows].squeeze())
            
            all_outputs.append(outputs)
           
    return all_outputs

def get_vector_from_vgg_model(modelpath,vectorType,chunks):
    model = get_vgg_model(modelpath)
    
    print('< -- get elmo model success! -- >')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_outputs = []
    
    for chunk in chunks:
        chunk = [preprocess(c).unsqueeze(0) for c in chunk]
        outputs = np.array([model(c).data.numpy().squeeze() for c in chunk])
        all_outputs.append(outputs)
        
    return all_outputs

def out_vec_by_word(sentencelist,label,all_outputs,matchfile,outpath,filetype,outword):
    ## merge all chunks
    sentencevectorlist = []
    outputslist = []
    for out in all_outputs:
        outlist = []
        for o in out:
            outlist.append(o)
        outputslist.extend(outlist)
    
    print('< -- output vectorlist success! -- >')
    ## output results
    if matchfile == None:
        for i in range(len(outputslist)):
            sentencevectorlist = outputslist[i]
            outpath = '../results/Vector_' + label + '_word_in_sentence_' + str(i)+ '.' + filetype
            savefiles(sentencelist[i].split(),sentencevectorlist,outpath,filetype,outword)
            
    else:
        matchf = readfiles(matchfile)
        for m in range(len(matchf)):
            idx = sentencelist[m].split().index(matchf[m])
            sentencevectorlist.append(outputslist[m][idx])            
        savefiles(matchf,sentencevectorlist,outpath,filetype,outword)
    
    return sentencevectorlist

def out_vec_by_row(sentencelist,all_outputs,outpath,filetype,outword):
    ## merge outputs
    outputslist = []
    sentencevectorlist = []
    for out in all_outputs:
        outputslist.extend(out)
    sentencevectorlist = [arr.flatten() for arr in outputslist]
    ## output results
    savefiles(sentencelist,sentencevectorlist,outpath,filetype,outword)
    
    return sentencevectorlist

def Text_vector(file, vectorType, label = 'gpt2_base',filetype = 'csv', outword = 'n', ifstpw=0, matchfile = None):            
    """
    	get vector of each row of text, if the word is not in this model, output nan 
     
    Parameters
    ----------
    file : file
        Text filepath
    label : str
        model label (unique singal for getting the model that you want to load)
    filetype : str, optional
        you can select outfile format {'txt','csv','xlsx'}, default format is csv
    outword : str, optional
        you can select if output words and vectors into one file {'n','y'}    
    ifstpw : bool
        whether delete words in stopwords {0,1} default is 0, do not delete words
    vectorType : str, optional
        you can select which type of vector you want {'word','sentence'} 
        --> 'sentence': default, output one vector of each row of data
        --> 'word': output vectors for each word of each row of data
    matchfile : str, optional
        
    Returns
    ----------
        output a file include Text(optional) and vectors
        
    """
    ## load data as list
    if isinstance(file, list):
        sentencelist = file
    else:
        sentencelist = readfiles(file)
    print('< -- read files as sentencelist success! -- >')
    
    ## divide text
    chunk_size = 500
    chunks = [sentencelist[i:i + chunk_size] for i in range(0, len(sentencelist), chunk_size)]

    ## get current model info
    modelinfo = get_modelInfo()
    if label in modelinfo['label'].values:
        modeltype = modelinfo.loc[modelinfo['label'] == label,'modeltype'].values[0]
        modelpath = modelinfo.loc[modelinfo['label'] == label,'modelpath'].values[0]
        dimension = int(modelinfo.loc[modelinfo['label'] == label,'dimension'].values[0])
    else:
        print("There is no such label in the models list, please add model info in models/modelinfo.csv")
    
    ## get outpath
    print('< -- get modelinformation success! -- >')
    ## output information
    if not os.path.exists('../results'):
        os.makedirs('../results')
    outpath = '../results/Vector_' + label + '.' + filetype
    
    ## load data file to list
    if vectorType == 'sentence':

        if modeltype == 'bert':
            all_outputs = get_vector_from_bert_model(modelpath,vectorType,chunks)
        
        elif modeltype == 'gpt2':
            all_outputs = get_vector_from_gpt2_model(modelpath,vectorType,chunks)
        
        elif modeltype == 'clip':
            all_outputs = get_vector_from_clip_model(modelpath,vectorType,chunks,multi='text')
        
        elif modeltype == 'glv' or modeltype == 'w2v' or modeltype == 'fast' or modeltype == 'cnt' or modeltype == 'rws':
            all_outputs = get_vector_from_pretrained_model(modeltype,modelpath,vectorType,chunks,dimension)
        
        elif modeltype == 'elmo':
            all_outputs = get_vector_from_elmo_model(modelpath,vectorType,chunks)
                                        
        else :
            print('please check if the modeltype is right:{"bert","glv","w2v","gpt2","clip","fast","rws","elmo","cnt"}')
        
        sentencevectorlist = out_vec_by_row(sentencelist,all_outputs,outpath,filetype,outword)
        
    elif vectorType == 'word' or vectorType == 'word_in_sentence':
                
        ## select different model
        if modeltype == 'bert':
            all_outputs = get_vector_from_bert_model(modelpath,vectorType,chunks)
        
        elif modeltype == 'gpt2':
            all_outputs = get_vector_from_gpt2_model(modelpath,vectorType,chunks)
        
        elif modeltype == 'clip':
            all_outputs = get_vector_from_clip_model(modelpath,vectorType,chunks,multi='text')
        
        elif modeltype == 'glv' or modeltype == 'w2v' or modeltype == 'fast' or modeltype == 'cnt' or modeltype == 'rws':
            all_outputs = get_vector_from_pretrained_model(modeltype,modelpath,vectorType,chunks,dimension)
        
        elif modeltype == 'elmo':
            all_outputs = get_vector_from_elmo_model(modelpath,vectorType,chunks)
                    
        else :
            print('please check if the modeltype is right:{"bert","glv","w2v","gpt2","clip","fast","rws","elmo","cnt"}')
        
        print('< -- get vector success! -- >')
        if vectorType == 'word_in_sentence':
            sentencevectorlist = out_vec_by_word(sentencelist,label,all_outputs,matchfile,outpath,filetype,outword)
        elif vectorType == 'word':
            sentencevectorlist = out_vec_by_row(sentencelist,all_outputs,outpath,filetype,outword)
           
    print('< -- results has output -- >')
    print(outpath)
    return sentencevectorlist

    
def Image_vector(imagefile, label = 'clip_base', filetype = 'csv', outword = 'n', vectorType = None):   
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
    
    modelinfo = get_modelInfo()
    if label in modelinfo['label'].values:
        modeltype = modelinfo.loc[modelinfo['label'] == label,'modeltype'].values[0]
        modelpath = modelinfo.loc[modelinfo['label'] == label,'modelpath'].values[0]
        dimension = int(modelinfo.loc[modelinfo['label'] == label,'dimension'].values[0])
    else:
        print("There is no such label in the models list, please add model info in models/modelinfo.csv")
    
    ## output information
    if not os.path.exists('results'):
        os.makedirs('results')
    outpath = 'results/Vector_' + label + '.' + filetype

    imagelist = readfiles(imagefile)
    image = [Image.open(f).convert("RGB") for f in imagelist]
    
    ## divide text
    chunk_size = 500
    chunks = [image[i:i + chunk_size] for i in range(0, len(image), chunk_size)]

    if modeltype == 'clip': 
        all_outputs = get_vector_from_clip_model(modelpath,vectorType,chunks,multi='image') 
    elif modeltype == 'vgg':
        all_outputs = get_vector_from_vgg_model(modelpath,vectorType,chunks) 
    else :
        print('please check if the modeltype is right:{"bert","glv","w2v","gpt2","clip","fast","rws","elmo","cnt"}')
    
    ## merge outputs
    sentencevectorlist = out_vec_by_row(imagelist,all_outputs,outpath,filetype,outword)
    print('< -- results has output -- >')
    
    return sentencevectorlist
