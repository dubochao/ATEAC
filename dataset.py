import re
from torchtext.legacy import data
import jieba
import logging
#import pyhanlp
jieba.setLogLevel(logging.INFO)
import torchtext
# stopword_file = './hit_stopwords.txt'

# 读取停用词列表
def get_stopword_list(file):
    with open(file, 'r', encoding='utf-8') as f:    #
        stopword_list = [word.strip('\n') for word in f.readlines()]
    return stopword_list
# 3. 导入停止词的语料库,
# 对文本进行停止词的去除
# stop = get_stopword_list(stopword_file)  # 获得停用词列表
def drop_stops(Jie_content, stop):
    # clean_content = []
    
    line_clean = []
    for line in Jie_content:
        if line in stop:
            continue
        line_clean.append(line)
    # clean_content.append(line_clean)
    return line_clean
def word_cut(text):
    text=str(text).lower().strip()
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
def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='csv', skip_header=True,
        train='train.csv', validation='test.csv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev
if __name__=='__main__':
    print(word_cut('这 发型 ， 我 咋 瞅着 有点 梦露 的 感觉\n ?!!！'))

