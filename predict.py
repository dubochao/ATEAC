import jieba
from net import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from Optimizer import *
attenTextCNN = model.attenTextCNN
GlobalAttention= model.GlobalAttention
LocalAttention=model.LocalAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path= 'net0/Our_bestcheckpoint.tar'
# Initialize / load checkpoint
if device=='cuda':
  checkpoint = torch.load(path ,map_location=lambda storage,loc: storage.cuda(0))
else:
  checkpoint = torch.load(path, map_location='cpu')

def predict_sentiment(net, vocab,sentence):
# """sentence是词语的列表"""
     device = list(net.parameters())[0].device
     sentence=jieba.lcut(sentence)
     print(sentence)
     while len(sentence)<6:
         sentence.append('<unk>')
     #print(sentence)
     sentence = torch.tensor([vocab.stoi[word]for word in sentence],
                             device=device)

     label = torch.argmax(net(sentence.view((1, -1))), dim=1)
     # torch.save({'global':out2.squeeze(0),'local':out1.squeeze(0)},'../x.pt')

     return 'positive' if label.item() == 0 else 'negative'
sentence='相当不错的酒店，房间等各方面都让人满意。'
# ['相当', '不错', '的', '酒店', '，', '房间', '等', '各', '方面', '都', '让', '人', '满意', '。']
#Very good hotel, rooms and other aspects are satisfactory.
model=checkpoint['model']
vocab=torch.load('data/ChnSentiCorp_htl_unba_10000/vocab.pt')['text_field'].vocab
#####mask  dnn
#word_cut=dataset.word_cut(sentence)
#input_ids = torch.tensor([vocab.stoi[word] for word in word_cut])
#token_type_ids区分上下句子 没多大用
#data =dict(input_ids=input_ids,\
#       token_type_ids=torch.zeros(input_ids.shape,dtype=torch.int32),\
#      attention_mask=input_ids.ge(2).int()
#      )
print(predict_sentiment(model,vocab,sentence))
