# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:59:19 2023

@author: Dell
"""
import os
import pandas as pd


def readfiles(filepath):
    filename = os.path.split(filepath)[1]
    filetype = os.path.splitext(filename)[1]
    if filetype == '.txt' or filetype == '.csv':
        with open(filepath,'r',encoding='utf-8') as f:
            text = f.read().strip('\n').split('\n')
                  
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
            df.to_csv(filepath,encoding='utf-8',header=0,index=False,na_rep='nan')    
        elif filetype=='csv':
            df.to_csv(filepath, encoding='utf_8_sig',header=0,index=False,na_rep='nan')
        elif filetype == 'xlsx':
            df.to_excel(filepath,encoding='utf-8',header=0,index=False,na_rep='nan')
    elif outword == 'y':
        if filetype=='txt':
            df.to_csv(filepath,encoding='utf-8',na_rep='nan')    
        elif filetype=='csv':
            df.to_csv(filepath, encoding='utf_8_sig',na_rep='nan')
        elif filetype == 'xlsx':
            df.to_excel(filepath,encoding='utf-8',na_rep='nan')


