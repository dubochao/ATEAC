import argparse
import torch
import torchtext.legacy.data as data
from torchtext.vocab import Vectors
import os
from net import lstm,RNN,CNN,GRU,CNN_MUI, CNN_GRU,CNN_LSTM,BiGRU,CNN_BiGRU,att_BILSTM,CNN_BiLSTM,model

import train
import dataset
import utils
from log import *
import sys
sys.path.append('./net')

args = utils.parse_args()
def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    print(f'There are {len(vectors.itos)} words in the vocabulary')
    return vectors



# 下载地址： https://pan.baidu.com/s/1oObY4A_Ovo1CY00UrgbBKg?_at_=1636877536302
# 数据概览： 10000 多条酒店评论数据，7000 多条正向评论，3000 多条负向评论
# 推荐实验： 情感/观点/评论 倾向性分析
# 数据来源：携程网
# 原数据集： ChnSentiCorp-Htl-unba-10000，由 谭松波 老师整理的一份数据集
# 加工处理：
# 将原来 1 万个离散的文件整合到 1 个文件中
# 将负向评论的 label 从 -1 改成 0
# path='./data/ChnSentiCorp_htl_unba_10000'
path='./data/online_shopping_10_cats'
# path='./data'
def load_dataset(text_field, label_field, args, **kwargs):

    # train_dataset, dev_dataset = dataset.get_dataset('data', text_field, label_field)

    train_dataset, dev_dataset = dataset.get_dataset(path, text_field, label_field)
    #args.seq_len = len(dev_dataset)
    if os.path.exists(path+'/vocab.pt'):
        pass
    else:
        if args.static and args.pretrained_name and args.pretrained_path:
            vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
            text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
        else:
            text_field.build_vocab(train_dataset, dev_dataset)
        label_field.build_vocab(train_dataset, dev_dataset)
    train_iter = data.Iterator(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, sort_within_batch=False, repeat=False)
    dev_iter = data.Iterator(dataset=dev_dataset, batch_size=args.batch_size, shuffle=True, sort_within_batch=False, repeat=False)
    # train_iter, dev_iter = data.Iterator.splits(
    #     (train_dataset, dev_dataset),
    #     batch_sizes=(args.batch_size, len(dev_dataset)),
    #     sort_key=lambda x: len(x.text),
    #     **kwargs)
    return train_iter, dev_iter  # 每个batch对应的词向量

logging.info('Loading data...')
# print('Loading data...')
if os.path.exists(path+'/vocab.pt'):
    iterdict = torch.load(path+'/vocab.pt')
    text_field, label_field = iterdict['text_field'], iterdict['label_field']
    train_iter, dev_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
else:

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    # TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=tokenize, lower=True, init_token='<sos>',
    #                                    eos_token='<eos>', pad_token='<pad>', unk_token='<unk>', fix_length=10)
    # text_field = data.Field(lower=True,include_lengths=True, fix_length=args.fix_length)
    train_iter, dev_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
    torch.save({'text_field': text_field, 'label_field': label_field}, path+'/vocab.pt')
args.vocabulary_size = len(text_field.vocab)
# print(TEXT.vocab.vectors.shape)
# word_vec = TEXT.vocab.vectors[TEXT.vocab.stoi['我']]词转为数字，数字转为词，词转为词向量
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
print('args.device != -1 and torch.cuda.is_available()',args.cuda)
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
logging.info('Parameters:')
# print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

model_classes = {
    # 'BiGRU': BiGRU.BiGRU(args),
    # 'LSTM': lstm.LSTM(args),
    # 'RNN': RNN.RNN(args),
    'TextCNN': CNN.CNN(args),
    'cnn_bigru': CNN_BiGRU.CNNBiGRU(args),
    #
    'cnn_bilstm': CNN_BiLSTM.CNN_BiLSTM(args),
    'cnn_mul': CNN_MUI.CNN_MUI(args),
    # 'GRU': GRU.GRU(args),
    'CNNGRU': CNN_GRU.CGRU(args),
    'CNNLSTM': CNN_LSTM.CLSTM(args),
    # #
    'attBILSTM':att_BILSTM.attBiLSTM(args),
    'Our':model.attenTextCNN(args),

}
try:
#    j=1
    for i in model_classes.keys():
#        while j <=512:
#            j=j*2
#            args.batch_size=j
#            train_iter, dev_iter = load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
        # print('\nRunning model {}...\n'.format(i))
            train.train(train_iter, dev_iter, model_classes[i],i, args)
except KeyboardInterrupt:
    logging.info('Exiting from training early')
