# -*- coding: utf-8 -*-
"""DATA_MR.ipynb

"""

#英文数据集 MR
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
#import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
#from gensim.models import word2vec
import time
import random
import nltk
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score
from stanfordcorenlp import StanfordCoreNLP

num_epochs = 200
embed_size = 300  #embedding_dim
#num_hiddens = 128#128
#num_layers = 2#2
#bidirectional = True#False
#batch_size = 16#非交叉：13，26，41，52，82  #交叉：13，26，41，82
labels = 2
#lr = 0.2
#pos_size = vocab_size
pos_dim = 25
tag_dim = 30
nltk.download('averaged_perceptron_tagger')


#MR   读取数据路径，标注分类类别
#创建Hownet 词list
def Tag_wordslist(filepath):
    Tag_words = [line.strip() for line in open(filepath,'r',encoding = 'utf-8').readlines()]
    return Tag_words

Adverbswords = Tag_wordslist('drive/python3/HowNet/Adverbs(E).txt')#加载程度副词的路径
NegativeRewords = Tag_wordslist('drive/python3/HowNet/NegativeRe(E).txt')#加载负面评价词的路径
NegativeSenwords = Tag_wordslist('drive/python3/HowNet/NegativeSen(E).txt')#加载负面情感词的路径
PositiveRewords = Tag_wordslist('drive/python3/HowNet/PositiveRe(E).txt')#加载正面评价词的路径
PositiveSenwords = Tag_wordslist('drive/python3/HowNet/PositiveSen(E).txt')#加载正面情感词的路径
Inverwords = Tag_wordslist('drive/python3/HowNet/Inver(E).txt')#加载否定情感词的路径
#stopwords = Tag_wordslist('F:\\yuliao\\Chinese_tag\\stopwords_sum.txt')#加载否定情感词的路径
def readCOAE_Train():
    data = []
    files= open("drive/python3/MU_BiLstm/train.txt", "r",encoding = 'utf-8')#drive/python3/COAE/train/pos.txt
    lines = files.readlines()
    for line in lines:
        if line[0] =='0':
            data.append([line[2:],0])
        else :
            data.append([line[2:],1])
    return data
def readCOAE_Test():
    data = []
    files= open("drive/python3/MU_BiLstm/test.txt", "r",encoding = 'utf-8')
    lines = files.readlines()
    for line in lines:
        if line[0] =='0':
            data.append([line[2:],0])
        else :
            data.append([line[2:],1])   
    return data

train_data = readCOAE_Train()
test_data = readCOAE_Test()
random.shuffle(train_data)
random.shuffle(test_data)

print(train_data)
print(test_data)
print(len(train_data))
print(len(test_data))

def clean_text(text):
    text = text.translate(string.punctuation)
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"'m", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text
def StanfordCore_NLP(data):
    nlp = StanfordCoreNLP('drive/python3/MR/stanford-corenlp-full-2018-10-05')
    datas = []#记录单词
    pos_tags = []
    parsess = []
    for review, score in data:
        parses = []
        review = clean_text(review)
        par = nlp.dependency_parse(review)#句法
       
        for parse in par:
            parses.append(parse[0])
        parsess.append(parses)
        pos_tag = []#记录所有词性   
        postags = nltk.pos_tag(review.split())#词性
        for postag in postags:
            pos_tag.append(postag[1])
        pos_tags.append(pos_tag)         
        datas.append(review.split())#分词
    nlp.close()
    return datas,pos_tags,parsess
train_datas = []#记录单词
train_tag = []#记录所有词性
train_parserss = []#记录句法分析
test_datas = []#记录单词
test_tag = []#记录所有词性
test_parserss = []#记录句法分析
train_datas,train_tag,train_parserss = StanfordCore_NLP(train_data)
test_datas,test_tag,test_parserss = StanfordCore_NLP(test_data)

tr_te_word = train_datas+test_datas  #word 词 后加
vocab = set(chain(*tr_te_word))
vocab_size = len(vocab) 

tr_te_par = train_parserss+test_parserss #句法依存
vocab_par = set(chain(*tr_te_par))
vocab_par_size = len(vocab_par)

#wvmodel = gensim.models.KeyedVectors.load_word2vec_format('drive/python3/MU_BiLstm/glove.6B.50d.txt',binary=False, encoding='utf-8')  #50b英文 
#wvmodel = gensim.models.KeyedVectors.load_word2vec_format('drive/python3/glove_model.txt',binary=False, encoding='utf-8')  #100b
wvmodel = gensim.models.KeyedVectors.load_word2vec_format('drive/python3/yuliao/glove_300b_model.txt',binary=False, encoding='utf-8')  #300b英文 
#wvmodel = gensim.models.KeyedVectors.load_word2vec_format('drive/python3/MU_BiLstm/glove.refine.txt',binary=False, encoding='utf-8')  #300b清华训练的
wordsList = np.load('drive/python3/yuliao/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() 
wordsList = [word.decode('UTF-8') for word in wordsList]
#wordsTag = np.load('drive/python3/COAE/wordsTag_30b_1.2.npy')#词性向量：情感词*1.2，非情感词*1.0
#wordsTag = np.load('drive/python3/wordsTag_30b.npy')#词性向量:情感词随机，非情感词为0

#5
num_epochs = 200
embed_size = 300  #embedding_dim
num_hiddens =128#128
num_layers = 2#2
bidirectional = True#False
batch_size = 25#25#非交叉：13，26，41，52，82  #交叉：13，26，41，82
word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'

par_to_idx = {word: i+1 for i, word in enumerate(vocab_par)}
par_to_idx['<unk>'] = 0
idx_to_par= {i+1: word for i, word in enumerate(vocab_par)}
idx_to_par[0] = '<unk>'

embed_word = nn.Embedding(vocab_size+2,embed_size) #对于未登录词，进行均匀分布（-0.1，0.1）之间随机初始化
embed_word.weight.data.uniform_(-0.05, 0.05)

weight = torch.zeros(vocab_size+3, embed_size)

embedd_tag = nn.Embedding(vocab_size+3,tag_dim) #词性
embedd_tag.weight.data.uniform_(-1.0, 1.0)

tag = torch.zeros(vocab_size+3, tag_dim)
for i in range(len(word_to_idx)):#词
    indstr = idx_to_word[i]
    indid = word_to_idx[idx_to_word[i]]
    if indstr in wordsList:
        weight[indid, :] = torch.from_numpy(wvmodel.get_vector(indstr))
    else:
        weight[indid, :] = embed_word.weight[indid]#未登录词        
    if indstr in Adverbswords or indstr in NegativeRewords or indstr in NegativeSenwords or indstr in PositiveRewords or indstr in PositiveSenwords or indstr in Inverwords:
        tag[indid, :] = 1.2*embedd_tag.weight[indid]
    else:
        tag[indid, :] = 0.01*embedd_tag.weight[indid]
weight[vocab_size+1, :] = embed_word.weight[vocab_size+1]#未登录词  后加   
tag[vocab_size+1, :] = 1.2*embedd_tag.weight[vocab_size+1]
tag[vocab_size+2, :] = 0.01*embedd_tag.weight[vocab_size+2]  
  
pos_size = vocab_size
pos_dim = 25
tag_size = vocab_size
tag_dim = 30
parser_size = vocab_size
parser_dim = 25
device = torch.device('cuda:0')
maxlen = 200
def encode_samples(datas,parserss,maxlen=60):#词特征、位置特征、词性
    features = []
    Lists = []
    Li_tags = []
    Li_pars = []
    for sample in datas:#sample为一条评论文本
        feature = []
        List = []
        Li_tag = []
        for token in sample:#token为一条评论文本里的一个单词
            if token in word_to_idx:
                feature.append(word_to_idx[token])
                Li_tag.append(word_to_idx[token])
            #位置特征向量    
                p = sample.index(token)-len(sample)+maxlen+1
                List.append(p)
            else:
                feature.append(vocab_size+1)#feature.append(vocab_size+1) 
                if token in Adverbswords or token in NegativeRewords or token in NegativeSenwords or token in PositiveRewords or token in PositiveSenwords or token in Inverwords:
                    Li_tag.append(vocab_size+1)
                else:
                    Li_tag.append(vocab_size+2)
               # feature.append(0)
               # Li_tag.append(0)
                List.append(0)
        features.append(feature)
        Lists.append(List)
        Li_tags.append(Li_tag)
    for sample in parserss:#句法
        Li_par = []
        for token in sample:#token为一条评论文本里的一个单词
            if token in par_to_idx:
                Li_par.append(par_to_idx[token])
            else:
                Li_par.append(0)
        Li_pars.append(Li_par) 
    return features,Lists,Li_tags,Li_pars  
def pad_samples(features, maxlen=35, PAD=0):
    padded_features = []
    count = 0
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
            
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features
  
def pad_list(Lists, maxlen=35, PAD=0):
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

def pad_tag(tags, maxlen=35, PAD=0):
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
def pad_par(parsers, maxlen=35, PAD=0):#句法
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
  
train_fea,train_lis,train_tag,train_pars = encode_samples(train_datas,train_parserss)#再传入词性
train_features = torch.LongTensor(pad_samples(train_fea))#词特征ID
train_Lists = torch.LongTensor(pad_list(train_lis))#位置特征ID
train_Tag = torch.LongTensor(pad_tag(train_tag))  #词性特征ID
train_parser = torch.LongTensor(pad_par(train_pars))  #句法
train_labels= torch.LongTensor([score for _, score in train_data])

test_fea,test_lis,test_tag,test_pars = encode_samples(test_datas,test_parserss)
test_features = torch.LongTensor(pad_samples(test_fea))
test_Lists = torch.LongTensor(pad_list(test_lis))
test_Tag = torch.LongTensor(pad_tag(test_tag))  #词性特征ID
test_parser = torch.LongTensor(pad_par(test_pars))  #句法
test_labels= torch.LongTensor([score for _, score in test_data])



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
    
#BiLSTM

print(len(train_data))
print(len(test_data))

class Model3(nn.Module):
    def __init__(self,labels,vocab_size,embedding_dim,weight,tag,hidden_dim,pos_size,pos_dim,batch,num_layers,tag_size,tag_dim,parser_size,parser_dim):  
        super(Model3,self).__init__()
        self.batch_size = batch
        self.embedding_dim_1 = embedding_dim  #100 model_1
       # self.embedding_dim_2 = embedding_dim_tag  #130 model_2
        self.hidden_dim = hidden_dim  
        self.vocab_size = vocab_size  
        self.labels = labels
        self.num_layers = num_layers
        self.pos_size = pos_size  
        self.pos_dim = pos_dim  #25
        self.tag_size = tag_size  
        self.tag_dim = tag_dim  #30
        self.parser_size = parser_size
        self.parser_dim = parser_dim
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1#2
        
        self.embedding_1 = nn.Embedding.from_pretrained(weight)
        self.embedding_1.weight.requires_grad = False
        self.tag_embeds = nn.Embedding.from_pretrained(tag)#weight_tag为7*30
        self.tag_embeds.weight.requires_grad = False
        self.pos_embeds = nn.Embedding(self.pos_size,self.pos_dim)#位置特征 pos_dim :25维 
        self.parser_embeds = nn.Embedding(self.parser_size,self.parser_dim)  #依存分析树25（随机）
        self.lstm_1 = nn.LSTM(input_size=self.embedding_dim_1+self.tag_dim,#词+词性
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#125b
        self.lstm_2 = nn.LSTM(input_size=self.embedding_dim_1+self.pos_dim,#词+位置
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#130b
        self.lstm_3 = nn.LSTM(input_size=self.embedding_dim_1+self.parser_dim,#词+句法
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers, 
                              bidirectional=self.bidirectional,
                              dropout=0,
                              batch_first = True
                             )#125b
        self.ln = LayerNorm(self.hidden_dim * self.num_directions)
        #self.logistic = nn.Linear(self.hidden_dim * self.num_directions*3,self.labels)
        self.logistic = nn.Linear(self.hidden_dim * self.num_directions*3,self.hidden_dim)
        self.logistic_1 = nn.Linear(self.hidden_dim,self.labels)
        self.dropout=nn.Dropout(p=0.5)
        self.dropout_emb=nn.Dropout(p=0.3)
        
        self._init_weights()
          
    def _init_weights(self, scope=1.):
        self.pos_embeds.weight.data.uniform_(-scope,scope)#parser_embeds
        self.parser_embeds.weight.data.uniform_(-scope,scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self):
        num_layer = self.num_layers * self.num_directions
        weight = next(self.parameters()).data
        return (Variable(weight.new(num_layer, self.batch_size, self.hidden_dim).zero_()), Variable(weight.new(num_layer, self.batch_size, self.hidden_dim).zero_()))

    def forward(self,sentence,pos,tag,paerser):#加tag
       # encode = self.embedding_1(sentence)
       # self._init_weights()
        encode_1 = torch.cat(((self.embedding_1(sentence)),(self.tag_embeds(tag))),2) #词向量+词性 拼接 330
        encode_2 = torch.cat(((self.embedding_1(sentence)),(self.pos_embeds(pos))),2) #词向量+位置 拼接 325
        encode_3 = torch.cat(((self.embedding_1(sentence)),(self.parser_embeds(paerser))),2) #词向量+句法 拼接 325
        #encoding_1 = self.dropout_emb(encode)#对拼接后的层dropout
        
        encode_1 =  self.dropout(encode_1)#Droupout下
        encode_2 =  self.dropout(encode_2)#Droupout下
        encode_3 =  self.dropout(encode_3)#Droupout下
        
        lstm_out_1, hidden = self.lstm_1(encode_1)
        lstm_out_2, hidden = self.lstm_2(encode_2)
        lstm_out_3, hidden = self.lstm_3(encode_3)
        
        lstm_out_1 = torch.transpose(lstm_out_1,0,1)
        lstm_out_2 = torch.transpose(lstm_out_2,0,1)
        lstm_out_3 = torch.transpose(lstm_out_3,0,1)
        
        lstm_out_1 = F.tanh(lstm_out_1)
        lstm_out_2 = F.tanh(lstm_out_2)
        lstm_out_3 = F.tanh(lstm_out_3)
        
       # lstm_out_1 =  self.dropout(lstm_out_1)#Droupout下
        #lstm_out_2 =  self.dropout(lstm_out_2)#Droupout下
        #lstm_out_3 =  self.dropout(lstm_out_3)#Droupout下
        
        output_1 = self.ln(lstm_out_1)[-1]
        output_2 = self.ln(lstm_out_2)[-1]
        output_3 = self.ln(lstm_out_3)[-1]
        
        output_1 = F.tanh(output_1)
        output_2 = F.tanh(output_2)
        output_3 = F.tanh(output_3)
        
        output = torch.cat([output_1,output_2,output_3],1)
       # output =  self.dropout(output)##Droupout下
        output = self.logistic(output)#jia
        output = F.relu(output)#jia
        output =  self.dropout(output)##Droupout下
        return F.log_softmax(self.logistic_1(output))

#1
device = torch.device('cuda:0')
#model= Model_1(labels,vocab_size+1,embed_size,embed_size_tag,weight,weight_tag,num_hiddens,pos_size,pos_dim,batch_size,num_layers)
model= Model3(labels,vocab_size+1,embed_size,weight,tag,num_hiddens,pos_size,pos_dim,batch_size,num_layers,tag_size,tag_dim,parser_size,parser_dim )
print(model)
model.to(device)
#torch.nn.utils.clip_grad_norm(model.parameters(), 10)

#model = model.cuda()
loss_function = nn.CrossEntropyLoss()# size_average默认情况下是True，对每个小批次的损失取平均值。 但是，如果字段size_average设置为False
#optimizer = optim.SGD(model.parameters(), lr=0.01,weight_decay=3e-8)
#optimizer = optim.SGD(model.parameters(), lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = optim.Adadelta(model.parameters(), lr=0.01, rho=0.9, eps=1e-06, weight_decay=0.001)
#optimizer = optim.Adadelta(model.parameters(), lr=0.01, rho=0.9, eps=1e-06, weight_decay=0.001)#Adam
optimizer = optim.Adagrad(model.parameters(), lr=0.01,  weight_decay=1e-6)
#optimizer = optim.Adam(model.parameters(), lr=0.01,eps=1e-06, weight_decay=0.0001)#Adam
train_set = torch.utils.data.TensorDataset(train_features,train_Lists,train_Tag,train_parser,train_labels)
test_set = torch.utils.data.TensorDataset(test_features,test_Lists,test_Tag,test_parser,test_labels)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)

for epoch in range(num_epochs+200):
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
        x_train_wvm = Variable(train_i[0].cuda())#.cuda()
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

