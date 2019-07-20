# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:01:23 2019

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
from nltk.stem import WordNetLemmatizer       #词形变化
from nltk import pos_tag
from nltk.corpus import wordnet
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
    
#aggregate the number of keywords of each paper
def sum_keywords(paper_dict):
    keywords_list = []
    for paper_name in paper_dict.keys():
#        print(paper_name)
#        print(paper_dict[paper_name]['ego_attribute']['keywords'])
        keywords_list.extend(paper_dict[paper_name]['ego_attribute']['keywords'])
    return dict(Counter(keywords_list).most_common())

def merge_key(keywords_dict, csv_keyword_path):
    with open(csv_keyword_path, "a", newline='') as f:
        for word, fre in keywords_dict.items():
                writer = csv.writer(f)
                writer.writerow((word,fre))
        f.close()    

def merge_keywords(adjust_dict):
    zip_list = []
    #all of the keywords
    label_sum = list(adjust_dict.keys())
    for index, item1 in enumerate(label_sum):
        candidate = []
        if index < 20:
                index = 20
        for item2 in label_sum[index-20:index+20]:
            if difflib.SequenceMatcher(None, item1, item2).quick_ratio() > 0.8:
                candidate.append(item2)
        if len(candidate) > 1 :
            print("%s\n"%(item1))
            print(*candidate,sep = '\n')
            i = int(input("Enter your input: "))-1
            zip_list.append(candidate[i])
        else:
            zip_list.append(candidate[0])
        print('\n\n\n\n')
    keywords_list = zip(label_sum, zip_list)
    keywords_list_order = deepcopy(sorted(keywords_list))
    return keywords_list_order

def last_word(keyword, wnl):
    key_word_list = keyword.split()
    last_word = key_word_list[-1]
    last_word = wnl.lemmatize(last_word, 'n')
    key_word_list[-1] = last_word    
    if last_word == 'estimator' :
        last_word = 'estimation'
        key_word_list[-1] = last_word
    if last_word == 'trial':
        last_word = 'experiment'
        key_word_list[-1] = last_word
    if last_word =='theory':
        del key_word_list[-1]
    keyword = ' '.join(key_word_list)
    return keyword
    
def keyword_dict(keywords_dict):
    key_dict = {}
    keywords_list = list(keywords_dict.keys())
    wnl = WordNetLemmatizer()
    re_pattern = re.compile(r'([(].*?[)])', re.S)
    for keyword in keywords_dict.keys():
        keyword_name = deepcopy(keyword)
        try:
            if ')' in keyword:
                keyword = keyword.replace(re.findall(re_pattern, keyword)[0],'')
            if '-' in keyword:
                keyword_ = last_word(re.sub('-', '', keyword), wnl)
                if keyword_ in keywords_list:
                    keyword_c = keyword_
                else:
                    keyword_c = last_word(re.sub('-', ' ', keyword), wnl)
            else:
                keyword_c = last_word(keyword, wnl)
        except:
            keyword_c = keyword
            print(keyword)
            i = int(input("Enter your input: "))-1
        if keyword_c != keyword:
            print((keyword,keyword_c))
        key_dict[keyword_name] =  keyword_c.replace('.', '').replace('\'s', '')
    return key_dict
            
def change_keyword(paper_dict, change_dict):
    for key in paper_dict.keys():
        keyword_list = []
        for keyword in paper_dict[key]['ego_attribute']['keywords']:
            keyword_list.append(change_dict[keyword])
        paper_dict[key]['ego_attribute']['keywords'] = keyword_list
    return paper_dict

def fre_1_from_keywords_dict(keywords_dict):
    keywords_dict_c = deepcopy(keywords_dict)
    circle = deepcopy(list(keywords_dict_c.keys()))
    for keyword in circle:
        if keywords_dict[keyword] < 3:
            del keywords_dict_c[keyword]
    return keywords_dict_c

if __name__ == "__main__":
    os.chdir("D:/bigdatahw/pan_paper")
    path_paper_dict_pkl = 'D:/bigdatahw/pan_paper/潘老师/pkl/paper_dict.pkl'
    csv_keyword_path = 'D:/bigdatahw/pan_paper/潘老师/csv/keywords.csv'
    paper_dict = read_pkl(path_paper_dict_pkl)
    keywords_dict = sum_keywords(paper_dict)
    change_dict = keyword_dict(keywords_dict)
    paper_dict = change_keyword(paper_dict, change_dict)
    keywords_dict = sum_keywords(paper_dict)
    
    #merge keywords
    merge_key(keywords_dict, csv_keyword_path)
    
    #读取csv
    adjust_dict = {}
    with open(csv_keyword_path,mode='r',encoding='ISO-8859-1',newline='') as csv_file:
        csv_reader_lines = csv.reader(csv_file)       #逐行读取csv文件
        next(islice(csv_reader_lines,0), None)   
        for one_line in csv_reader_lines:                          
            adjust_dict[one_line[0]] = one_line[1].rstrip()
    merge_key(adjust_dict, csv_keyword_path)  
    
    keywords_dict = fre_1_from_keywords_dict(keywords_dict)
    
    #构造文章和词汇二部图
    df_article_word = pd.DataFrame(index = paper_dict.keys(), columns = keywords_dict.keys()).fillna(0) 
    for article in paper_dict.keys():
        for keyword in paper_dict[article]['ego_attribute']['keywords']:
            df_article_word.loc[article,keyword] += 1
    keywords_list = list(keywords_dict.keys())
    keywords_list_c = deepcopy(keywords_list)
    index_tuple = []
    for index1 in keywords_list:
        for index2 in keywords_list_c:
            paper_array = df_article_word[index1] + df_article_word[index2]
            com_label = paper_array[paper_array==2]._stat_axis.values.tolist()
            com_label_cou = len(com_label) 
            index_tuple.append([index1,index2,com_label_cou])
        keywords_list_c.remove(index1)  
        print(index1)
        
    #筛选出连边大于1的
    index_tuple_true = [item for item in index_tuple if item[0]!='']
    index_tuple_true = [item for item in index_tuple_true if item[1]!='']
    index_tuple_true = [item for item in index_tuple_true if item[2] > 2]
    index_tuple_csv = [item for item in index_tuple_true if item[0] != item[1]]
    index_node_csv = [(item[0],item[2]) for item in index_tuple_true if item[0] == item[1]]
    
    
    #写入csv
    csv_tuple_path = 'D:/bigdatahw/pan_paper/潘老师/csv/keywords_tuple_2.csv'
    with open(csv_tuple_path, "a", newline='') as f:
        for tuple_fre in index_tuple_csv:
                writer = csv.writer(f)
                writer.writerow(tuple_fre)
        f.close()  
    
    #写入csv
    csv_node_path = 'D:/bigdatahw/pan_paper/潘老师/csv/keywords_node_2.csv'
    with open(csv_node_path, "a", newline='') as f:
        for tuple_fre in index_node_csv:
                writer = csv.writer(f)
                writer.writerow(tuple_fre)
        f.close()      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    