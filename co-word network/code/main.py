# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 01:19:30 2019

@author: Bokkin Wang
"""
import sys
import os
sys.path.append("D:/bigdatahw/pan_guan/code")
import ldamodel
from ldamodel import *

os.chdir("D:/bigdatahw/pan_guan")
# 设定num_topics的值为主题数量，period有两个选项 '_18_09'和'_08_98'。代表两个年份
train_lda(period = '_08_98', num_topics = 4, remove_top = 4) 


