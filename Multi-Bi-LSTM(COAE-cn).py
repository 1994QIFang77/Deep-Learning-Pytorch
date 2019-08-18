# -*- coding: utf-8 -*-
"""DATA_COAE.ipynb
"""

COAE：中文数据集

#中文数据处理：
#1、数据（去停用词）、分词、标词性、句法分析，
#2、对未登录词进行均匀分布随机化
#3、train与test同时统计

#词向量、词性向量、位置向量、句法结构
#中文数据集 COAE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import nltk
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score
import pyltp
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
num_epochs = 200  #训练迭代次数
embed_size = 300  #词向量维度
num_hiddens = 128 #隐藏层神经单元个数
num_layers = 2    #隐藏层层数
bidirectional = True  #False为LSTM,True为双向LSTM
labels = 2          #二分类

#Hownet  （词性）
def Tag_wordslist(filepath):
    Tag_words = [line.strip() for line in open(filepath,'r',encoding = 'utf-8').readlines()]
    return Tag_words
CAdverbswords = Tag_wordslist('drive/python3/COAE/Chinese_tag/Adverbs(C).txt')#加载程度副词的路径
CNegativeRewords = Tag_wordslist('drive/python3/COAE/Chinese_tag/NegativeRe(C).txt')#加载负面评价词的路径
CNegativeSenwords = Tag_wordslist('drive/python3/COAE/Chinese_tag/NegativeSen(C).txt')#加载负面情感词的路径
CPositiveRewords = Tag_wordslist('drive/python3/COAE/Chinese_tag/PositiveRe(C).txt')#加载正面评价词的路径
CPositiveSenwords = Tag_wordslist('drive/python3/COAE/Chinese_tag/PositiveSen(C).txt')#加载正面情感词的路径
CInverwords = Tag_wordslist('drive/python3/COAE/Chinese_tag/Inver(C).txt')#加载否定情感词的路径


#COAE   读取数据路径，标注分类类别
def readCOAE_Pos():
    data = []
    files= open("drive/python3/COAE/COAE6千不分/pos_1.txt", "r",encoding = 'utf-8')  #drive/python3/COAE/train/pos.txt
    lines = files.readlines()
    for line in lines:
        data.append([line,1])
    return data
def readCOAE_Neg():
    data = []
    files= open("drive/python3/COAE/COAE6千不分/neg_-1.txt", "r",encoding = 'utf-8')
    lines = files.readlines()
    for line in lines:
        data.append([line,0])   
    return data

Pos = readCOAE_Pos()
Neg = readCOAE_Neg()
random.shuffle(Pos) #随机打乱数据
random.shuffle(Neg) #随机打乱数据
train_data = []
test_data = []
train_data = Pos[664:]+Neg[655:]
test_data = Pos[:664]+Neg[:655]
random.shuffle(train_data)
random.shuffle(test_data)

#合工大LTP分词、词性、句法分析
def LTP(data):
    LTP_DATA_DIR = 'drive/python3/COAE/ltp_data_v3.4.0'       #ltp模型目录的路径
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    #加载词性相关文件
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    #加载句法分析相关文件
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    datas = []#记录单词
    pos_tag = []#记录所有词性
    parserss = []#记录句法分析
    for review, score in data:
        #pyltp分词
        words = segmentor.segment(review)  # 分词
        words= " ".join(words).split()
        data = []
        for word in words:
            #if word not in stopwords:#去停用词
            data.append(word)    
        #pylt词性标注 
        postags = postagger.postag(data)  # 词性标注
        postags= ' '.join(postags).split()
        pos_tag.append(postags)
                #依存句法分析
        arcs = parser.parse(data, postags)  # 句法分析
        parsers = []
        for arc in arcs:
            parsers.append(arc.relation)   
            
        datas.append(data) 
        parserss.append(parsers)
    segmentor.release()  # 释放模型 
    postagger.release() 
    parser.release()  
    return datas,pos_tag,parserss

train_datas = []#记录单词
train_tag = []#记录所有词性
train_parserss = []#记录句法分析

train_datas,train_tag,train_parserss = LTP(train_data)

#词
tr_te_word = train_datas
vocab_word = set(chain(*tr_te_word)) #对所有词进行去重
vocab_word_size = len(vocab_word) #统计所以词的个数(不计重复)
word_to_idx = {word: i+1 for i, word in enumerate(vocab_word)}  #对去重后的所有词进行重新整理，标注序号，word：id
word_to_idx['<unk>'] = 0  
idx_to_word = {i+1: word for i, word in enumerate(vocab_word)}  #id：word
idx_to_word[0] = '<unk>'

#句法依存
tr_te_par = train_parserss
vocab_par = set(chain(*tr_te_par))
vocab_par_size = len(vocab_par)
par_to_idx = {word: i+1 for i, word in enumerate(vocab_par)}
par_to_idx['<unk>'] = 0
idx_to_par= {i+1: word for i, word in enumerate(vocab_par)}
idx_to_par[0] = '<unk>'

#加载词向量，300维
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('drive/python3/COAE/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5',binary=False)  #300b中文 
print("end")
wordsList = np.load('drive/python3/COAE/wordsList_Chinese.npy') #加载已经整理好的词List 索引
wordsList = wordsList.tolist() 
embed_size_word = 300  #词向量维度
embed_size_tag = 30   #词性向量维度
weight_word = torch.zeros(vocab_word_size+3, embed_size_word)   #根据COAE数据集中每个词的id，重新生成的词向量数组 
weight_tag = torch.zeros(vocab_word_size+3, embed_size_tag)     #词性向量数组，同上

embed_word = nn.Embedding(vocab_word_size+2,embed_size_word) #对于未登录词，进行均匀分布（-0.1，0.1）之间随机初始化
embed_word.weight.data.uniform_(-0.1, 0.1)

embedd_tag = nn.Embedding(vocab_word_size+3,embed_size_tag) #随机生成一个词性向量，生成范围在（-0.1,0.1）之间
embedd_tag.weight.data.uniform_(-1.0, 1.0)

for i in range(len(word_to_idx)):
    indstr = idx_to_word[i]
    indid = word_to_idx[idx_to_word[i]]
    if indstr in wordsList:
        weight_word[indid, :] = torch.from_numpy(wvmodel.get_vector(indstr))
    else:
        weight_word[indid, :] = embed_word.weight[indid]#未登录词        
    if indstr in CAdverbswords or indstr in CNegativeRewords or indstr in CNegativeSenwords or indstr in CPositiveRewords or indstr in CPositiveSenwords or indstr in CInverwords:
        weight_tag[indid, :] = 1.2*embedd_tag.weight[indid]#根据HowNet，来重点标注词（重点词乘以1.2）
    else:
        weight_tag[indid, :] = embedd_tag.weight[indid]
weight_word[vocab_word_size+1, :] = embed_word.weight[vocab_word_size+1]#未登录词  
weight_tag[vocab_word_size+1, :] = 1.2*embedd_tag.weight[vocab_word_size+1]
weight_tag[vocab_word_size+2, :] = embedd_tag.weight[vocab_word_size+2]

pos_size = vocab_word_size+3
pos_dim = 25
tag_size = vocab_word_size+3
tag_dim = 30
parser_size = vocab_word_size+3
parser_dim = 25
batch_size = 16 

#将COAE数据集中每一条样例，进行词语与ID 转化
def encode_samples(train_datas,train_tag,train_parserss,maxlen=161):#词特征、位置特征、词性、句法
    features = []#词向量
    Lists = []#位置
    Li_tags = []#词性
    Li_pars = []#句法
    for sample in train_datas:#词向量、位置、词性
        feature = [] 
        List = []
        Li_tag = []#词性
        c = 161
        for token in sample:#token为一条评论文本里的一个单词
            if token in word_to_idx:
                feature.append(word_to_idx[token])#词
                Li_tag.append(word_to_idx[token])#词性
                p = sample.index(token)-len(sample)+maxlen+1  #位置特征向量
                List.append(p)
            else:
                feature.append(vocab_word_size+1) 
                if token in CAdverbswords or token in CNegativeRewords or token in CNegativeSenwords or token in CPositiveRewords or token in CPositiveSenwords or token in CInverwords:
                    Li_tag.append(vocab_word_size+1)
                else:
                    Li_tag.append(vocab_word_size+2)
             
                List.append(0)
                
        features.append(feature)
        Lists.append(List)
        Li_tags.append(Li_tag)

    for sample in train_parserss:#句法
       Li_par = []
       for token in sample:#token为一条评论文本里的一个单词
            if token in par_to_idx:
                Li_par.append(par_to_idx[token])
            else:
                Li_par.append(0)
       Li_pars.append(Li_par) 
    return features,Lists,Li_tags,Li_pars   

def pad_samples(features, maxlen=50, PAD=0):#词，每一条样例中，词语个数只保留前50个词，长度大于50直接去掉，小于50，以0填充
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
            
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features
  
def pad_list(Lists, maxlen=50, PAD=0):#位置，与词一样
    padded_Lists = []  #位置ID
    for List in Lists:
        if len(List) >= maxlen:
            padded_List = List[:maxlen]
        else:
            padded_List = List
            while(len(padded_List) < maxlen):
                padded_List.append(PAD)
        padded_Lists.append(padded_List)
    return padded_Lists

def pad_tag(tags, maxlen=50, PAD=0):#词性，一样
    padded_Tag = []  #词性ID
    for tag in tags:
        if len(tag) >= maxlen:
            padded_t = tag[:maxlen]
        else:
            padded_t = tag
            while(len(padded_t) < maxlen):
                padded_t.append(PAD)
        padded_Tag.append(padded_t)
    return padded_Tag
 
def pad_par(parsers, maxlen=50, PAD=0):#句法，一样
    padded_Par = []  #词性ID
    for tag in parsers:
        if len(tag) >= maxlen:
            padded_p = tag[:maxlen]
        else:
            padded_p = tag
            while(len(padded_p) < maxlen):
                padded_p.append(PAD)
        padded_Par.append(padded_p)
    return padded_Par

#词向量、位置、词性、句法
train_fea,train_lis,train_tag,train_pars = encode_samples(train_datas,train_tag,train_parserss)#再传入词性
train_features = torch.LongTensor(pad_samples(train_fea))#词特征ID
train_Lists = torch.LongTensor(pad_list(train_lis))#位置特征ID
train_Tag = torch.LongTensor(pad_tag(train_tag))  #词性特征ID
train_parser = torch.LongTensor(pad_par(train_pars))  #句法
train_labels= torch.LongTensor([score for _, score in train_data])#标签

#层归一化
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
       # self.hidden_size = hidden_size*3
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(self.hidden_size))
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)

#Bi-LSTM 
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,labels,embedding_dim,embedding_dim_tag,weight,tag,hidden_dim,pos_size,pos_dim,batch,num_layers,tag_size,tag_dim,parser_size,parser_dim):  
        super(Model,self).__init__()
        self.batch_size = batch #batch_size批次大小
        self.embedding_dim_1 = embedding_dim  #词向量维度 300b
     
        self.hidden_dim = hidden_dim  #隐藏神经单元个数 (128)
      
        self.labels = labels          #标签大小 2
        self.num_layers = num_layers  #隐藏层个数
        self.pos_size = pos_size      #位置特征行 pos_size = vocab_size
        self.pos_dim = pos_dim        #位置特征维度 25 
        self.tag_size = tag_size      #词性特征行 tag_size = vocab_size
        self.tag_dim = tag_dim        #词性特征维度 30 
        self.parser_size = parser_size  #句法特征行 parser_size = vocab_size
        self.parser_dim  = parser_dim   #句法特征维度 25
        self.bidirectional = True    #False为LSTM True为Bilstm
        self.num_directions = 2 if self.bidirectional else 1 
        self.embedding_1 = nn.Embedding.from_pretrained(weight) #引入词向量
        self.embedding_1.weight.requires_grad = False
        
        self.tag_embeds = nn.Embedding.from_pretrained(tag) #引入词性向量30(对情感词，否定词，副词*1.2权值)
        self.tag_embeds.weight.requires_grad = False
 
        self.pos_embeds = nn.Embedding(self.pos_size,self.pos_dim) #位置特征25（随机） 
        
        self.parser_embeds = nn.Embedding(self.parser_size,self.parser_dim)  #依存分析树25（随机）
      
        self.lstm_1 = nn.LSTM(input_size=self.embedding_dim_1+self.pos_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#125b（词+位置）
        
        self.lstm_2 = nn.LSTM(input_size=self.embedding_dim_1+self.tag_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#130b（词+词性）
        
        self.lstm_3 = nn.LSTM(input_size=self.embedding_dim_1+self.parser_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#125b(词+依存分析)
        self.ln = LayerNorm(self.hidden_dim * self.num_directions) 
        self.Linear = nn.Linear(self.hidden_dim * self.num_directions*3,self.hidden_dim) #线性层768-》128
        self.logistic_1 = nn.Linear(self.hidden_dim,self.labels)#128-》2
        self.dropout=nn.Dropout(p=0.5)
        self.dropout_emb=nn.Dropout(p=0.3)
        self._init_weights()
    
    #初始化：pos_embeds、parser_embeds      
    def _init_weights(self, scope=1.):
        self.pos_embeds.weight.data.uniform_(-scope,scope)
        self.parser_embeds.weight.data.uniform_(-scope,scope)
 
    def forward(self,sentence,pos,tag,parse):
        embeds_1 = torch.cat(((self.embedding_1(sentence)),(self.pos_embeds(pos))),2) #词向量+位置 拼接 325
        embeds_2 = torch.cat(((self.embedding_1(sentence)) , (self.tag_embeds(tag))),2)  #词向量+词性 拼接 330
        embeds_3 = torch.cat(((self.embedding_1(sentence)) , (self.parser_embeds(parse))),2) #词向量+句法 拼接 325
 
        states_1, hidden = self.lstm_1(embeds_1)#Bi-LSTM
        states_2, hidden = self.lstm_2(embeds_2)#Bi-LSTM
        states_3, hidden = self.lstm_3(embeds_3)#Bi-LSTM
        
        #转置
        states_1 = torch.transpose(states_1,0,1)
        states_2 = torch.transpose(states_2,0,1)
        states_3 = torch.transpose(states_3,0,1)
        
        #tanh激励函数
        states_1 = F.tanh(states_1)
        states_2 = F.tanh(states_2)
        states_3 = F.tanh(states_3)
 
        #层归一化
        states_1 = self.ln(states_1)[-1]
        states_2 = self.ln(states_2)[-1]
        states_3 = self.ln(states_3)[-1]
       
        #tanh激励函数
        states_1 = F.tanh(states_1)
        states_2 = F.tanh(states_2)
        states_3 = F.tanh(states_3)
        
        output = torch.cat([states_1,states_2,states_3],1) #将三个通道结果，进行拼接，融合
        output = self.Linear(output)#线性层 768-》128
        output = F.relu(output)#relu激励
        return F.log_softmax(self.logistic_1(output))#128->2

#1
device = torch.device('cuda:0')
model= Model(labels,embed_size,embed_size_tag,weight_word,weight_tag,num_hiddens,pos_size,pos_dim,batch_size,num_layers,tag_size,tag_dim,parser_size,parser_dim)
print(model)
model.to(device)
loss_function = nn.CrossEntropyLoss()# size_average默认情况下是True，对每个小批次的损失取平均值。 但是，如果字段size_average设置为False
optimizer = optim.Adagrad(model.parameters(), lr=0.01,  weight_decay=0.001)
train_set = torch.utils.data.TensorDataset(train_features,train_Lists,train_Tag,train_parser,train_labels)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)

test_datas = []#记录单词
test_tag = []#记录所有词性
test_parserss = []#记录句法分析
test_datas,test_tag,test_parserss = LTP(test_data)

test_fea,test_lis,test_tag,test_pars = encode_samples(test_datas,test_tag,test_parserss)
test_features = torch.LongTensor(pad_samples(test_fea))
test_Lists = torch.LongTensor(pad_list(test_lis))
test_Tag = torch.LongTensor(pad_tag(test_tag))  #词性特征ID
test_parser = torch.LongTensor(pad_par(test_pars))  #句法
test_labels= torch.LongTensor([score for _, score in test_data])

test_set = torch.utils.data.TensorDataset(test_features,test_Lists,test_Tag,test_parser,test_labels)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=True)

num_epochs = 140#94
for epoch in range(num_epochs):
    print("epoch:%d" %epoch)
    acc=0 
    total=0
    running_loss=0.0
    test_losses=0.0
    n = 0.0
    m= 0.0
    model.train()
    for train_i in train_iter:
        n += 1
        x_train_wvm = Variable(train_i[0].cuda())
        x_train_pos = Variable(train_i[1].cuda())
        x_train_tag = Variable(train_i[2].cuda())
        x_train_parser = Variable(train_i[3].cuda())
        label = Variable(train_i[4].cuda())
        y = model(x_train_wvm,x_train_pos,x_train_tag,x_train_parser) #词向量+位置特征向量
        loss = loss_function(y, label)
        optimizer.zero_grad() #将参数的grad值初始化为0
        loss.backward()#
        optimizer.step()    
         
        y = np.argmax(y.cpu().data.numpy(),axis=1)
        running_loss += float(loss)
        for y1,y2 in zip(y,label):
            if y1==y2:
                acc+=1
            total+=1
    #print(n)
    #print(len(train_iter.dataset))
    print("loss:%.5f,train:%.4f %%" %(float(running_loss)/n,(100*float(acc)/total)))
 
   #acc_t = 0
   # total_t = 0
    count_predict = [0,0]
    count_total = [0,0]
    count_right = [0,0]
    test_acc = 0.0
    #with torch.no_grad():  # no grad when test and predict
    model.eval()
    #hidden = model.init_hidden()
    m = 0
    for test_i in test_iter:
        m += 1
       
        x_test_wvm = Variable(test_i[0].cuda())
        x_test_pos = Variable(test_i[1].cuda())
        x_test_tag = Variable(test_i[2].cuda())
        x_test_parser = Variable(test_i[3].cuda())
        test_label = Variable(test_i[4].cuda())
        
        yy = model(x_test_wvm,x_test_pos,x_test_tag,x_test_parser)
        
        test_loss = loss_function(yy, test_label)
        test_losses += test_loss
        y = np.argmax(yy.cpu().data.numpy(),axis=1)
        
        test_acc += accuracy_score(torch.argmax(yy.cpu().data,
                                                    dim=1), test_label.cpu())
        for y1,y2 in zip(y,test_label):
            count_predict[y1]+=1
            count_total[y2]+=1
            if y1==y2:
                count_right[y1]+=1

    #print(count_predict)
    #print(count_total)
   # print(count_right)
    precision = [0,0]
    recall = [0,0]
    for i in range(len(count_predict)):
        if count_predict[i]!=0:
            precision[i] = float(count_right[i])/count_predict[i]
            recall[i] = float(count_right[i])/count_total[i]
    
    precision = sum(precision)/len(precision)
    recall = sum(recall)/len(recall)    

    print("loss:%.5f , 准确率：%.3f , 召回率：%.3f , f：%.3f   | %.3f" %(test_losses.data/m,precision,recall,(2*precision*recall)/(precision+recall),test_acc/m))
