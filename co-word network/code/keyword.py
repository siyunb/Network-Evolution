# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:50:29 2019

@author: Bokkin Wang
"""
import os
from math import log, sqrt
import re
import sys
import csv
sys.path.append("D:/bigdatahw/pan_guan/code")
import difflib
from copy import deepcopy
from itertools import islice   
import pandas as pd
import pickle as pkl
import random
import warnings
from collections import Counter
warnings.filterwarnings("ignore")

#读取序化模型
def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

#写出序化模型
def write_pkl(path_pkl,abtract_complete):
    abtract_complete_file = open(path_pkl, 'wb')
    pkl.dump(abtract_complete, abtract_complete_file)
    abtract_complete_file.close()

#set path
def set_path():
    #### the preparation of workplace################################################
    global path_data_pkl_18_09, path_data_pkl_08_98, lda_path,\
         dictionary_path_18_09, dictionary_path_08_98, \
         corpus_path_18_09, corpus_path_08_98, model_path_18_09, \
         model_path_08_98, journal_path_18_09, journal_path_08_98, \
         keyword_path_18_09, keyword_path_08_98, keyword_plus_path_18_09, \
         keyword_plus_path_08_98, key_path_18_09, key_path_08_98, key_path,\
         keyword_path, csv_keyword_path, content_18_09_path, content_08_98_path,\
         content_path
         
    journal_path_18_09 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'journal_dict_18_09.pkl'
    journal_path_08_98 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'journal_dict_08_98.pkl'
    path_data_pkl_18_09 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'abtract_18_09.pkl'
    path_data_pkl_08_98 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'abtract_08_98.pkl'
    lda_path = os.getcwd() + os.sep + 'lda' 
    dictionary_path_18_09 = lda_path + os.sep + 'dictionary'+ os.sep +'dictionary_18_09.dictionary' 
    dictionary_path_08_98 = lda_path + os.sep + 'dictionary'+ os.sep +'dictionary_08_98.dictionary' 
    corpus_path_18_09 = lda_path + os.sep + 'corpus' + os.sep +'corpus_18_09.mm' 
    corpus_path_08_98 = lda_path + os.sep + 'corpus' + os.sep +'corpus_08_98.mm' 
    model_path_18_09 = lda_path + os.sep + 'model' + os.sep + 'lda_18_09.pkl'
    model_path_08_98 = lda_path + os.sep + 'model' + os.sep + 'lda_08_98.pkl'
    keyword_path_18_09 =  os.getcwd() + os.sep + 'pkl' + os.sep+ 'keyword_dict_18_09.pkl'
    keyword_path_08_98 =  os.getcwd() + os.sep + 'pkl' + os.sep+ 'keyword_dict_08_98.pkl'
    keyword_plus_path_18_09 =  os.getcwd() + os.sep + 'pkl' + os.sep+ 'keyword_plus_dict_18_09.pkl'
    keyword_plus_path_08_98 =  os.getcwd() + os.sep + 'pkl' + os.sep+ 'keyword_plus_dict_08_98.pkl'
    key_path_18_09 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'key_dict_18_09.pkl'
    key_path_08_98 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'key_dict_08_98.pkl'
    key_path = os.getcwd() + os.sep + 'pkl' + os.sep+ 'key_dict.pkl'
    keyword_path = os.getcwd() + os.sep + 'pkl' + os.sep+ 'keyword_dict.pkl'
    csv_keyword_path = os.getcwd() + os.sep + 'csv' + os.sep+ 'keyword.csv'
    content_18_09_path = os.getcwd() + os.sep + 'pkl' + os.sep+ 'content_18_09.pkl'
    content_08_98_path = os.getcwd() + os.sep + 'pkl' + os.sep+ 'content_08_98.pkl'
    content_path = os.getcwd() + os.sep + 'pkl' + os.sep+ 'content_08_98.pkl'
    
    
#there are two opinions for 'dictionary', journals_dict_18_09 and journals_dict_08_98,respectively,  
#and two opinions for 'char',keyword_stat and keyword_plus_stat.
def union_dict_for_word(dictionary, char, path):
    store_dict = {}
    for journal_name in dictionary.keys():
        keywords = set(sum([list(obj[char].keys()) for obj in dictionary[journal_name].values()] , []))       
        while '' in keywords :
            keywords.remove('')
        total = {}    
        for key in keywords:
            total[key] = sum([obj[char].get(key , 0) for obj in dictionary[journal_name].values()])
        store_dict[journal_name] = deepcopy(total)
    write_pkl(path,store_dict)

def union_dict(*objs):
    keys = set(sum([list(obj.keys()) for obj in objs] , []))
    total = {}
    for key in keys:
        total[key] = sum([obj.get(key , 0) for obj in objs])
    return total

def union_dict_(objs, path):
    keys = set(sum([list(obj.keys()) for obj in objs] , []))
    total = {}
    for key in keys:
        total[key] = sum([obj.get(key , 0) for obj in objs])
    write_pkl(path,total)
    
def keyword_and_plus(keyword,plus,path):
    keyword_dict = {}
    for journal_name in keyword.keys():
        keyword_dict[journal_name] = deepcopy(union_dict(keyword[journal_name], plus[journal_name]))
    write_pkl(path,keyword_dict)

def adjust_dictionary(keyword_dict_ori, adjust_dict):
    keyword_dict = deepcopy(keyword_dict_ori)
    #adjust keyword_dict
    keyword_list = list(keyword_dict.keys())
    for keyword1 in keyword_list:
        if adjust_dict[keyword1] == keyword1:
            continue
        else:
            keyword2 = deepcopy(adjust_dict[keyword1])
            if keyword2 in keyword_list:
                keyword_dict[keyword2] += keyword_dict[keyword1]
                keyword_dict.pop(keyword1)
            else:
                keyword_dict[keyword2] = keyword_dict[keyword1]
                keyword_dict.pop(keyword1)
    return keyword_dict

def adjust_period_dict(key_dict_ori, adjust_dict):
    key_dict = deepcopy(key_dict_ori)
    journal_list = list(key_dict.keys())
    for journal in journal_list:
        ad = adjust_dictionary(key_dict[journal], adjust_dict)
        key_dict[journal] = deepcopy(ad)
    return key_dict 

def keyword2abstrct(key_dict):
    content_dict = []
    for journal, value in key_dict.items():
        content = []
        for word,i in key_dict[journal].items():
            content.extend([word]*i)    
        random.shuffle(content)    
        content_dict.append(content)
    return content_dict

def keyword2abstrct_(content_18_09, content_08_98):
    content_dict = []
    for i in range(len(content_18_09)):
        content_dict.append(content_18_09[i] + content_08_98[i])   
    random.shuffle(content_dict)    
    return content_dict

##删除词频为1的词汇
def remove_flu_one(abtract_complete):
    abtract_complete_count = []
    for onelist in abtract_complete:
        abtract_complete_count.extend(onelist)
    word_count = (Counter(abtract_complete_count).most_common())
    word_count_top = word_count[0:20]                            #词频最高的10个单词
    for i in range(len(word_count)-1, -1, -1):
        if word_count[i][1] == 1:
            word_count_top.append(word_count[i])                #删除次品唯一的单词
    for word_list in abtract_complete:
        for minute_word in word_count_top:
            while minute_word[0] in word_list:
                word_list.remove(minute_word[0])
    return abtract_complete

if __name__ == "__main__":
    os.chdir("D:/bigdatahw/pan_guan")
    set_path()
    journals_dict_18_09 = read_pkl(journal_path_18_09)
    journals_dict_08_98 = read_pkl(journal_path_08_98)
    
    #store the keywords from different years through pkl
#    union_dict_for_word(journals_dict_18_09, 'keyword_stat', keyword_path_18_09)
#    union_dict_for_word(journals_dict_18_09, 'keyword_plus_stat', keyword_plus_path_18_09)
#    union_dict_for_word(journals_dict_08_98, 'keyword_stat', keyword_path_08_98)
#    union_dict_for_word(journals_dict_08_98, 'keyword_plus_stat', keyword_plus_path_08_98)
    
    #generate the stat of keyword and keyword_plus 
    keyword_dict_18_09 = read_pkl(keyword_path_18_09)
    keyword_dict_08_98 = read_pkl(keyword_path_08_98)
    keyword_plus_dict_18_09 = read_pkl(keyword_plus_path_18_09)
    keyword_plus_dict_08_98 = read_pkl(keyword_plus_path_08_98)
    
    #read the keyword
    key_dict_18_09 = read_pkl(key_path_18_09)
    key_dict_08_98 = read_pkl(key_path_08_98)
    key_dict = read_pkl(key_path)
    
    #merge keyword and keyword_plus and store
#    keyword_and_plus(keyword_dict_18_09, keyword_plus_dict_18_09, key_path_18_09)
#    keyword_and_plus(keyword_dict_08_98, keyword_plus_dict_08_98, key_path_08_98)
    keyword_and_plus(key_dict_18_09, key_dict_08_98, key_path)

    
    #combination of all keywords
#    union_dict_(key_dict.values(), keyword_path)
    keyword_dict = read_pkl(keyword_path)
    
    #merge keywords
    keywords_key = list(set(keyword_dict.keys()))
    keywords_value = list(keyword_dict.values())
    keywords_list = zip(keywords_key, keywords_value)
    keywords_list_order = deepcopy(sorted(keywords_list))
    for i in keywords_list_order:
        with open(csv_keyword_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(i)
    f.close()
    
    #读取csv
    adjust_dict = {}
    with open(csv_keyword_path,mode='r',encoding='ISO-8859-1',newline='') as csv_file:
        csv_reader_lines = csv.reader(csv_file)       #逐行读取csv文件
        next(islice(csv_reader_lines,0), None)   
        for one_line in csv_reader_lines:                          
            adjust_dict[one_line[0]] = one_line[1].rstrip()

    #adjust dict
    keyword_dict_ad = adjust_dictionary(keyword_dict, adjust_dict)
    key_dict_18_09_ad = adjust_period_dict(key_dict_18_09, adjust_dict)
    key_dict_08_98_ad = adjust_period_dict(key_dict_08_98, adjust_dict)
    
    #keyword to abstract
    content_18_09 = keyword2abstrct(key_dict_18_09)
    content_08_98 = keyword2abstrct(key_dict_08_98)
    content = keyword2abstrct_(content_18_09, content_08_98)           
    
    #write
    write_pkl(content_18_09_path,content_18_09)            
    write_pkl(content_08_98_path,content_08_98)            
    write_pkl('D:\\bigdatahw\\pan_guan\\pkl\\key_dict_18_09_ad.pkl',key_dict_18_09_ad)            
    write_pkl('D:\\bigdatahw\\pan_guan\\pkl\\key_dict_08_98_ad.pkl',key_dict_08_98_ad)             
    write_pkl(content_path,content) 
    
    #read
    for journal, value in key_dict_18_09.items():
        csv_path = 'D:\\bigdatahw\\pan_guan\\csv\\' + journal + '.csv'
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            for word, num in value.items():               
                writer.writerow([word, num])
            f.close()            
    