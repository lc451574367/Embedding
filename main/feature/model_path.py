# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:36:03 2023

Module Help:
    add_modelInfo
    get_modelInfo
    show_modelInfo
    
@author: Cheng Liu

"""
import os
import pandas as pd
allmodelinformation = 'feature/models/modelInfo.csv'
defaultmodelinformation = 'feature/models/modelInfo_default.csv'
 
def add_modelInfo(label, modeltype, modelpath,modal = None, modelname=None, corpus = None, layer = None, dimension = None, vocasize = None, language = None, official = None, download = None, reference = None):
    defaultinfo = pd.read_csv(allmodelinformation)
    newmodel = {'label':label,'modal':modal,'modeltype':modeltype ,'modelname':modelname,'modelpath': modelpath,'corpus':corpus,'layer':layer,'dimension':dimension,'Vocabulary size':vocasize,'language':language,'official':official,'download address':download,'reference':reference}
    data = pd.DataFrame([newmodel])
    if label in defaultinfo['label'].tolist():
        print("The label name has already exists!!")
    else:    
        modelInformationTable = pd.concat([defaultinfo,data],ignore_index=True)
        modelInformationTable.to_csv(allmodelinformation,index=False)
        show_modelInfo(allmodelinformation)

def delete_modelInfo(label):
    modelInformationTable = pd.read_csv(defaultmodelinformation)
    modelInformationTable = modelInformationTable[modelInformationTable['label'] !=label]
    modelInformationTable.to_csv(allmodelinformation,index=False)
    show_modelInfo(allmodelinformation)
    
    
def get_modelInfo(modelpath=allmodelinformation):
    modelInformationTable = pd.read_csv(modelpath)
    return modelInformationTable

def show_modelInfo(modelpath=allmodelinformation):
    modelInformationTable = get_modelInfo(modelpath)
    print(modelInformationTable.to_markdown())
    