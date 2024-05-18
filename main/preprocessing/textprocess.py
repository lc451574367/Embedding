# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:59:19 2023

@author: Dell
"""
import os
import pandas as pd
import string

def readfiles(filepath):
    filename = os.path.split(filepath)[1]
    filetype = os.path.splitext(filename)[1]
    if filetype == '.txt' or filetype == '.csv':
        with open(filepath,'r',encoding='utf-8') as f:
            text = f.read()
            text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            text = text.strip('\n').split('\n')
                  
    elif filetype == '.xlsx':
        df = pd.read_excel(filepath,header=None)
        value = df.values[:,0]
        text = value.tolist()
    return text

def savefiles(word, vector, filepath, filetype='csv', outword='n'):
    df = pd.DataFrame(vector)
    df.index = word
    
    if outword == 'n':
        if filetype=='txt':
            df.to_csv(filepath,encoding='utf-8',header=False,index=False,na_rep='nan')    
        elif filetype=='csv':
            df.to_csv(filepath, encoding='utf_8_sig',header=False,index=False,na_rep='nan')
        elif filetype == 'xlsx':
            df.to_excel(filepath,encoding='utf-8',header=False,index=False,na_rep='nan')
    elif outword == 'y':
        if filetype=='txt':
            df.to_csv(filepath,encoding='utf-8',na_rep='nan',index=True,header=False)    
        elif filetype=='csv':
            df.to_csv(filepath, encoding='utf_8_sig',na_rep='nan',index=True,header=False)
        elif filetype == 'xlsx':
            df.to_excel(filepath,encoding='utf-8',na_rep='nan',index=True,header=False)

def readfiles_in_directory(directory_path):
    # 存储文件名和内容的字典
    file_contents = {}

    # 遍历指定目录下的所有文件
    for file in os.listdir(directory_path):
        # 构建文件的完整路径
        file_path = os.path.join(directory_path, file)
        filename = os.path.splitext(file)[0]
        # 确保是文件而不是目录
        if os.path.isfile(file_path):
            # 
            file_contents[filename] = readfiles(file_path)
    return file_contents
