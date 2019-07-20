# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 23:47:18 2018

@author: Bokkin Wang
"""

import os
import json
import re
import csv 
import difflib
import numpy as np
from copy import deepcopy
import pandas as pd
from itertools import islice   
import sys
sys.path.append("D:/bigdatahw/coauthor network/data processing")
from time_test import time_transform
from collections import Counter
import pickle as pkl

#读取一条原始论文信息，并抽取想要的信息
class journal_paper(object):           
    
    def __init__(self,one_line):
        self.authors = [author.split('@')[0].replace(', ',',').replace('  ',' ').replace('. ','.').title().replace('-','') for author in one_line[10].split('::')]  #创建一个作者名录
        self.title = ' '.join(one_line[0].lower().strip().replace('  ',' ').split())
        self.publisher = ' '.join(one_line[1].lower().strip().replace('  ',' ').split())
        self.doi = one_line[2].lower()
        self.publish_date = one_line[3].strip().replace('  ',' ')
        self.cited_num = one_line[4]
        self.time_cited = one_line[5]
        self.abstract = one_line[6]
        self.keyword = [item.strip().lower().replace('  ',' ') for item in one_line[7].split(',')]
        self.keyword_plus = [item.strip().lower().replace('  ',' ') for item in one_line[8].split(',')]
        self.university = list(set([thing for item in one_line[10].split('::') for thing in item.split('@')[1:]]))
        self.cite_papers = one_line[12].split('::')

#打开csv，并按行读取
def read_iterate(filepath):
    paper_title = []
    with open(filepath,mode='r',encoding='ISO-8859-1',newline='') as csv_file:
        csv_reader_lines = csv.reader(csv_file)       #逐行读取csv文件
        next(islice(csv_reader_lines,1), None)   
        for one_line in csv_reader_lines:                          
            if one_line[0].lower() not in paper_title:     #加入文档去重  
                paper_title.append(' '.join(one_line[0].lower().split()))                
                yield one_line

#选择需要年份的引用论文转化为想要的信息
def citepaper_to_citemassage(year,cite_papers):
    
    def string_process(string):
        string = string.strip()
        string = string.replace(', ',',')
        string = string.replace('  ',' ')
        string = string.replace('   ',' ')
        return string    
    
    def main_to(one_cite,year):
        try:
            if one_cite != '': 
                cite_paper_name = '+'.join(one_cite.split('+')[0:-3]).lower()
                if cite_paper_name != 'no title' :
                    cite_paper_authors = [string_process(item).title() for item in one_cite.split('+')[-3].split(';')]
                    cite_paper_journal = one_cite.split('+')[-2].lower()
                    cite_paper_date = time_transform(one_cite.split('+')[-1])  
                else:
                    cite_paper_name = 'no title'
                    cite_paper_authors = [string_process(item).title() for item in one_cite.split('+')[1].split(';')]
                    cite_paper_authors.remove('Et Al.')
                    cite_paper_journal = '+'.join(one_cite.split('+')[2:-1]).lower()
                    cite_paper_date = time_transform(one_cite.split('+')[-1]) 
                return [cite_paper_journal,cite_paper_name,cite_paper_authors,cite_paper_date,year]
            else:
                return []            
        except:
            return [] 
            print(one_cite)
            print(cite_paper_name)
    
    cite_massage = list(map(main_to, cite_papers,list(np.repeat(year, len(cite_papers)))))    
    while [] in cite_massage:
        cite_massage.remove([])
    return cite_massage

#将csv转化为节点信息    
def csv_to_nod(path_csv,path_nod,path_pkl,journal_attribute,year_list):
    csvnames = os.listdir(path_csv)
    journals_dict = {}
    for csvname in csvnames:
        journal_name = os.path.splitext(csvname)[0]
        journals_dict[journal_name.lower()] = {}
    journals_dict_list = [item.split('.')[0] for item in csvnames]
    i = 0
    for journalname in journals_dict_list: 
        filepath = path_csv+'/'+journalname+'.csv'  
        journalname = journalname.lower()
        for one_line in read_iterate(filepath): 
            try:
                year = time_transform(one_line[3]).split('-')[0]    #日期作为键值
                if year in year_list:    #可以更改所研究的范围
                    if year not in journals_dict[journalname].keys():
                        journals_dict[journalname][year] = deepcopy(journal_attribute)
                    one_paper = journal_paper(one_line)
                    journals_dict[journalname][year]['abstract_sum'] += one_paper.abstract
                    if 'Et Al.' in one_paper.authors:    
                        journals_dict[journalname][year]['author_list'].extend(one_paper.authors.remove('Et Al.'))
                    journals_dict[journalname][year]['cite_sum'] += int(one_paper.cited_num.replace(',',''))
                    journals_dict[journalname][year]['cited_time'] += int(one_paper.time_cited.replace(',',''))
                    journals_dict[journalname][year]['paper_name'].append(one_paper.title)
                    journals_dict[journalname][year]['paper_sum'] += 1
                    journals_dict[journalname][year]['keyword_plus_stat'].extend(one_paper.keyword_plus)   #关键词没有去重
                    journals_dict[journalname][year]['keyword_stat'].extend(one_paper.keyword)             #没有去重
                    cite_massages =  citepaper_to_citemassage(year,one_paper.cite_papers)
                    for cite_massage in cite_massages:
                        try:
                            cited_journal = cite_massage[0].lower()
                            if cited_journal not in journals_dict[journalname][year]['cite_concrete'].keys():                        
                                journals_dict[journalname][year]['cite_concrete'][cited_journal] = []
                            journals_dict[journalname][year]['cite_concrete'][cited_journal].append(cite_massage[1:4])
            #                if cited_journal not in journals_dict.keys():
            #                    journals_dict[cited_journal] = {}
            #                if cite_massage[-1] not in journals_dict[cited_journal].keys():
            #                    journals_dict[cited_journal][cite_massage[-1]] = deepcopy(journal_attribute) 
            #                if journalname not in journals_dict[cited_journal][cite_massage[-1]]['cited_concrete'].keys():
            #                    journals_dict[cited_journal][cite_massage[-1]]['cited_concrete'][journalname]= []
            #                journals_dict[cited_journal][cite_massage[-1]]['cited_concrete'][journalname].append(one_paper.title)                               
                        except:
                            pass
                    i = i+1
                    print(i)
            except:
                pass
            
    #写出csv
    for journal in journals_dict.keys(): 
        for year in journals_dict[journal].keys():
            journals_dict[journal][year]['author_list'] = dict(Counter(journals_dict[journal][year]['author_list']).most_common())
            journals_dict[journal][year]['keyword_plus_stat'] = dict(Counter(journals_dict[journal][year]['keyword_plus_stat']).most_common())
            journals_dict[journal][year]['keyword_stat'] = dict(Counter(journals_dict[journal][year]['keyword_stat']).most_common())    
#        one_journal = [journal,json.dumps(journals_dict[journal])]
#        with open(path_nod+'/journal_nod1.csv', "a", newline='') as f:
#            writer = csv.writer(f)
#            try:
#                writer.writerow(one_journal)
#                f.close()
#            except:
#                f.close()
                
    #写出序化模型
    journals_dict_file = open(path_pkl, 'wb')
    pkl.dump(journals_dict, journals_dict_file)
    journals_dict_file.close()    
    
    
            
if __name__ == '__main__':  
    journal_attribute = {'cite_sum':0,'cited_time':0,'cite_concrete':{},'cited_concrete':{},'abstract_sum':'',
                         'paper_sum':0,'author_list':[],'keyword_stat':[],'keyword_plus_stat':[],'country':[],
                         'concrete_massage':{},'paper_name':[]}            #设定journal属性
    #######################################################
    #######信息需要另外爬取，爬虫利用原来爬取杂志名的jcr爬虫######
    #######################################################
    os.chdir("D:/bigdatahw/pan_guan")
    path_csv = 'D:/bigdatahw/pan_guan/data'
    path_nod = 'D:/bigdatahw/pan_guan/nod'
    path_pkl = 'D:/bigdatahw/pan_guan/pkl/journal_dict_18_09.pkl'
#    year_list = ['2008','2007','2006','2005','2004','2003','2002','2001','2000','1999','1998']
    year_list = ['2018','2017','2016','2015','2014','2013','2012','2011','2010','2009']
    csv_to_nod(path_csv,path_nod,path_pkl,journal_attribute,year_list)               #开始生成节点
    #请注意这很重要因为不仅仅是杂志很有可能是已出版的图书
    

    
    
    
    
    