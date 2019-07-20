# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:55:13 2019

@author: Bokkin Wang
"""
import os
import math
import pyLDAvis
import pyLDAvis.gensim
from copy import deepcopy
import pickle as pkl
from gensim.models import LdaModel
from collections import Counter
from gensim import corpora, models
from pprint import pprint

#读取序化模型
def read_pkl(path_pkl):
    x = open(path_pkl, 'rb')
    journals_dict = pkl.load(x,encoding='iso-8859-1')
    x.close()
    return journals_dict

def set_path():
    #### the preparation of workplace################################################
    global path_data_pkl_18_09, path_data_pkl_08_98, lda_path,\
         dictionary_path, content_path, corpus_path, model_path, \
         corpus_path_18_09, corpus_path_08_98, model_path_18_09, \
         model_path_08_98, journal_path_18_09, journal_path_08_98, \
         keyword_path_18_09, keyword_path_08_98, keyword_plus_path_18_09, \
         keyword_plus_path_08_98, key_path_18_09, key_path_08_98, key_path,\
         keyword_path, csv_keyword_path, content_18_09_path, content_08_98_path
         
    journal_path_18_09 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'journal_dict_18_09.pkl'
    journal_path_08_98 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'journal_dict_08_98.pkl'
    path_data_pkl_18_09 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'abtract_18_09.pkl'
    path_data_pkl_08_98 = os.getcwd() + os.sep + 'pkl' + os.sep+ 'abtract_08_98.pkl'
    lda_path = os.getcwd() + os.sep + 'lda' 
    dictionary_path = lda_path + os.sep + 'dictionary'+ os.sep +'dictionary.dictionary'  
    corpus_path_18_09 = lda_path + os.sep + 'corpus' + os.sep +'corpus_18_09.mm' 
    corpus_path_08_98 = lda_path + os.sep + 'corpus' + os.sep +'corpus_08_98.mm'
    corpus_path = lda_path + os.sep + 'corpus' + os.sep +'corpus.mm'
    model_path_18_09 = lda_path + os.sep + 'model' + os.sep + 'lda_18_09.pkl'
    model_path_08_98 = lda_path + os.sep + 'model' + os.sep + 'lda_08_98.pkl'
    model_path = lda_path + os.sep + 'model' + os.sep + 'lda.pkl'
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

    
def create_dictionary(dictionary_path, content_18_09_path, content_08_98_path):
    abtract_complete1 = read_pkl(content_18_09_path)
    abtract_complete2 = read_pkl(content_08_98_path)
#    if not os.path.exists(dictionary_path):
#        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
    dictionary = corpora.Dictionary(abtract_complete1)
    dictionary.add_documents(abtract_complete2)
    dictionary.save(dictionary_path)  #permanent dictionary
#    else:
#        print('=== 检测到词典已经存在，跳过该阶段 ===')  
        
def create_corpus(dictionary, path_corpus, path_data, remove_top): 
    abtract_complete = read_pkl(path_data)
    if remove_top != 0:
        abtract_complete = deepcopy(remove_flu_one(abtract_complete, remove_top))
#    if not os.path.exists(path_corpus):
#        print('=== 未检测到有语料库存在，开始遍历生成语料库 ===')
    corpus = [dictionary.doc2bow(text) for text in abtract_complete]  
    corpora.MmCorpus.serialize(path_corpus, corpus)            
#    else:
#        print('=== 检测到语料库已经存在，跳过该阶段 ===') 

def convert_tfidf(path_corpus):
    corpus = corpora.MmCorpus(path_corpus)
    #转tfidf向量
    tfidf = models.TfidfModel(corpus)
    corpusTfidf = tfidf[corpus]
    return corpusTfidf

def train_lda(period = '_18_09', num_topics = 4, remove_top = 0, tfidf = False, passes = 40, iterations = 600, eval_every = None):
    set_path()
    create_dictionary(dictionary_path, content_18_09_path, content_08_98_path)
    dictionary = corpora.Dictionary.load(dictionary_path)
    create_corpus(dictionary, eval('corpus_path'+ period), eval('content'+ period+ '_path'), remove_top)
    corpus = corpora.MmCorpus(eval('corpus_path'+ period))
    temp = dictionary[0]
    id2word = dictionary.id2token
    
    if tfidf:
        corpusTfidf = convert_tfidf(eval('corpus_path'+ period))
        model = LdaModel(corpus=corpusTfidf, id2word=id2word, \
                   alpha='auto', eta='auto', \
                   iterations=iterations, num_topics=num_topics, \
                   passes=passes, eval_every=eval_every)
    else:
        #建立模型
        model = LdaModel(corpus=corpus, id2word=id2word, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, num_topics=num_topics, \
                           passes=passes, eval_every=eval_every)
    
    #序化模型
    lda_model_file = open(eval('model_path'+ period), 'wb')
    pkl.dump(model, lda_model_file)
    lda_model_file.close() 
    
    #输出每个主题的关键词
    top_topics = model.top_topics(corpus)
    print('每个主题的关键词:') 
    pprint(top_topics)    
    
    #每一行包含了主题词和主题词的权重
    print('前两个主题的主题词和主题词权重:')
    model.print_topic(0,10)
    model.print_topic(1,10)   
    
    #给训练集输出其属于不同主题概率   
    print('输出前十本杂志属于不同主题的概率:')    
    for i in list(range(10)):
        for index, score in sorted(model[corpus[i]], key=lambda tup: -1*tup[1]):
            print(index, score)
            
    #calculate perplexity
    testset = []
    for i in range(corpus.num_docs):
        testset.append(corpus[i])
    perplexity(model, testset, dictionary, len(dictionary.keys()), num_topics)
       
    #LDA visualization---------------------------------------------------        
    vis_wrapper = pyLDAvis.gensim.prepare(model,corpus,dictionary)
    pyLDAvis.display(vis_wrapper)
    pyLDAvis.save_html(vis_wrapper,"lda%dtopics.html"%num_topics)
    pyLDAvis.show(vis_wrapper)
    
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    prep = 0.0
    prob_doc_sum = 0.0
    topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        doc_word_num = 0 # the num of words in the doc
        for word_id, num in doc:
            prob_word = 0.0 # the probablity of the word 
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                # cal p(w) : p(w) = sumz(p(z)*p(w|z))
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
    print ("the perplexity of this ldamodel is : %s"%prep)
    return prep

def remove_flu_one(abtract_complete, remove_top):
    abtract_complete_count = []
    for onelist in abtract_complete:
        abtract_complete_count.extend(onelist)
    word_count = (Counter(abtract_complete_count).most_common())
    word_count_top = word_count[0:remove_top]                            
#    for i in range(len(word_count)-1, -1, -1):
#        if word_count[i][1] == 1:
#            word_count_top.append(word_count[i])                #删除次品唯一的单词
    abtract_complete_n = []
    for word_list in abtract_complete:
        for minute_word in word_count_top:
            while minute_word[0] in word_list:
                word_list.remove(minute_word[0])
        abtract_complete_n.append(deepcopy(word_list))        
    return abtract_complete_n

if __name__ == '__main__':
    os.chdir("D:/bigdatahw/pan_guan")
    train_lda()



    
    

    

    

  
        

