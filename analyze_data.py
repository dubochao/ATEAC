import jieba
import matplotlib.pyplot as plt
# import nltk
import torch
from tqdm import tqdm
import pandas as  pd
# from config import *
# from utils import normalizeString
import re
import numpy as np
plt.rcParams['font.sans-serif']=['KaiTi']
# 解决中文乱码问题
plt.rcParams['axes.unicode_minus']=False
# 解决负号无法显示的问题
def word_cut(text):
    text=str(text).lower().strip()
    # print(text)
    # exit()
    text = jieba.lcut(text)
    #split_content=''
    for i, j in enumerate(text):
        if re.search(r'[a-zA-z]+|[\u4e00-\u9fa5]|[？?！!]+|[0-9]+', j, re.M | re.I):
            pass
        else:
            text[i] = ''
            # del(a[i])
    while '' in text:
        text.remove('')
    #line = ''.join(regex .findall(text))
    #print(split_content.strip(' '))
    #clean_content=split_content
    # clean_content = drop_stops(Jie_content, stop)
    return text
def analyze_zh(biao):
    sent_lengths = []
    if 'text' in biao.keys():
        text = biao['text']
    else:
        text = biao['review']
    print()
    for sentence in tqdm(text):

        seg_list = word_cut(sentence)
        # Update word frequency

        sent_lengths.append(len(seg_list))

######################取众数#################################
    print('最小长度为：',min(sent_lengths))
    print('平均数：',np.mean(sent_lengths))
    max_value=np.bincount(sent_lengths).argmax()
    print('众数为：',sent_lengths[max_value])
    print('最值为：', max(sent_lengths))
    sent_lengths.sort(reverse=True)
    # autolabel(plt.bar(range(len(sent_lengths[:100])), sent_lengths[:100],fc='b'))
    num_bins = 50
    n, bins, patches =plt.hist(sent_lengths, num_bins, facecolor='#000000', alpha=0.5)

    # title = 'The distribution of the review length'
    title ='句子长度分布'
    plt.title(title)
    plt.xlabel('句子长度')
    plt.ylabel('句子数量')
    # plt.xlabel('Length')
    # plt.ylabel('Number')
    plt.show()



if __name__ == '__main__':

    #online_shopping_10_cats
    # b = pd.read_csv('./data/online_shopping_10_cats/online_shopping_10_cats.csv', index_col=None)
    b = pd.read_csv('./data/ChnSentiCorp_htl_unba_10000/data.csv', index_col=None)

    analyze_zh(b)
