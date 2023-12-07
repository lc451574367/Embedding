# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:37:44 2023

@author: Dell
"""
from model_path import *
from get_vector import *
from textprocess import *


"-----------------------------------------------1. test word---------------------------------------------"
wordpath = 'test/test.txt'

# file = wordpath
# vector = wordvectorlist
# model_path = glove_840B_300d
# dimension = 300
# df = pd.DataFrame(wordvectorlist)
model,size = get_glove_model_by_diff_para()

"entire models"
Word_vector(wordpath)
Word_vector(wordpath,modeltype='glove', corpus='common', dimension=300, size='M')
Word_vector(wordpath,modeltype='glove', corpus='wiki', dimension=300)
Word_vector(wordpath,modeltype='glove', corpus='wiki', dimension=200)
Word_vector(wordpath,modeltype='glove', corpus='wiki', dimension=100)
Word_vector(wordpath,modeltype='glove', corpus='wiki', dimension=50)
Word_vector(wordpath,modeltype='glove', corpus='twitter', dimension=200)
Word_vector(wordpath,modeltype='glove', corpus='twitter', dimension=100)
Word_vector(wordpath,modeltype='glove', corpus='twitter', dimension=50)
Word_vector(wordpath,modeltype='glove', corpus='twitter', dimension=25)
Word_vector(wordpath,modeltype='word2vec', corpus='GoogleNews')

"test parameter"
Word_vector(wordpath,modeltype='glove') # [glove_840B_300d_g2w] [default]
Word_vector(wordpath,modeltype='glove',corpus='common') # 
Word_vector(wordpath,modeltype='glove',corpus='wiki') # 
Word_vector(wordpath,modeltype='glove',corpus='twitter') # 
Word_vector(wordpath,modeltype='word2vec') # 


"-----------------------------------------------2. test sentence---------------------------------------------"
"test bert activate"
model_name = bert_L_12_H_256d_uncase
# bert默认端口为5555
command = 'bert-serving-start ' + '-model_dir '+ bert_path + '/' + model_name + '/' + ' -num_worker=1'
# 通过其他端口启动bert服务
command = 'bert-serving-start ' + '-model_dir '+ bert_path + '/' + model_name + '/' + ' -num_worker=1' + ' -port=5777 -port_out=5778'

# subprocess.call(command) # 没响应
# subprocess.run(command, shell=True) # 没响应
# subprocess.Popen(command) # 可用
"""启动、关闭bert服务"""
# 启动bert-service
p = subprocess.Popen(command) 
# 获取bert-service进程的端口号，并杀死该端口的进程
p.kill()
outinfo = subprocess.Popen('netstat -ano|findstr 5555', stdout=subprocess.PIPE, shell=True).communicate()
port = outinfo[0].decode().split()[-1]
subprocess.Popen(f'taskkill /F /PID {port}', shell=True)

# 杀死当前进程，并重启
# pid = os.getpid()
# os.kill(pid, signal.SIGTERM)
# os.popen(command) # 可用
# os.system(command) # 可运行，但输出占满了控制台不能进行后续的运行

# 客户端获取
bc = BertClient()
# 对应着端口号的客户端
bc1 = BertClient()

"测试函数"
sentencepath = 'test/sentence.txt'

file = sentencepath
sentencelist = readfiles(file)
sentencevectorarr = bc.encode(sentencelist)
sentencevectorarr = bc1.encode(sentencelist)
# vector = wordvectorlist
# model_path = glove_840B_300d
# dimension = 300
# df = pd.DataFrame(wordvectorlist)
# model = get_bert_model_by_diff_para(layer=12,dimension=768,case='uncase',corpustype='none')

from model_path import *
from get_vector import *
from textprocess import *
sentencepath = 'test/sentence.txt'
"""
每次启动bert只能访问一次，如需换模型进行访问，则需要关闭当前窗口，重新开启一个窗口运行
"""
Sentence_vector(sentencepath, modeltype = 'bert')
Sentence_vector(sentencepath, modeltype = 'bert', layer = 2)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 9) # wrong layer input
Sentence_vector(sentencepath, modeltype = 'bert', layer = 2, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 4, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 6, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 8, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 10, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 12, dimension=512)
Sentence_vector(sentencepath, modeltype = 'bert', layer = 2, dimension=300) # wrong dimension input
Sentence_vector(sentencepath, modeltype = 'bert', layer = 2, dimension=768,case='uncase') 
Sentence_vector(sentencepath, modeltype = 'bert', layer = 4, dimension=768,case='none') # wrong case input
Sentence_vector(sentencepath, modeltype = 'bert', layer = 8, dimension=768,case='uncase',corpustype='none') 
Sentence_vector(sentencepath, modeltype = 'bert', layer = 12, dimension=768,case='case',corpustype='111') # wrong corpustype input
Sentence_vector(sentencepath, modeltype = 'bert', layer = 24, dimension=1024,case='uncase',corpustype='none') 
Sentence_vector(sentencepath, modeltype = 'bert', layer = 24, dimension=768,case='uncase',corpustype='none') # wrong dimension input
Sentence_vector(sentencepath, modeltype = 'bert', layer = 24, dimension=768,case='case',corpustype='none') # wrong dimension input



Sentence_vector(sentencepath, modeltype = 'glove')
Sentence_vector(sentencepath, modeltype = 'glove', corpus='wiki',dimension = 1)

Sentence_vector(sentencepath, modeltype = 'word2vec', corpus='GoogleNews')














