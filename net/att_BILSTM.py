
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Neural Networks model : Bidirection LSTM
"""


class attBiLSTM(nn.Module):

    def __init__(self, args):
        super(attBiLSTM, self).__init__()
        self.args = args
        # attn_model=args.attn_model
        # class_num = args.class_num
        # filter_num = args.filter_num
        # seq_len=args.batch_size
        # filter_sizes = args.filter_sizes

        self.hidden_dim = args.filter_num
        self.num_layers = args.layers
        V =  args.vocabulary_size
        D =  args.embedding_dim
        C = args.class_num
        # self.embed = nn.Embedding(V, D)
        # pretrained  embedding
        self.embedding = nn.Embedding(V, D)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,
                                                            freeze=not args.non_static)  # non_static=FALSE 可以微调\s
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True,
                              bias=True)
        self.w_omega = nn.Linear(
            self.hidden_dim, self.hidden_dim,bias=False)
        self.u_omega = nn.Linear(self.hidden_dim, 1,bias=False)
        
        self.hidden2label2 = nn.Linear(self.hidden_dim, C)
        # self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        embed = self.embedding(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = torch.tanh(bilstm_out)
        Q = self.u_omega(M)
        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        newM  = torch.softmax(Q, dim=1)
        # Q = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        restoreM  = newM.squeeze(2)

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = torch.softmax(restoreM,dim=1)

        r = torch.bmm(bilstm_out.permute(0, 2, 1), torch.reshape(self.alpha, [-1, newM.size(1), 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = r.squeeze(2)

        sentenceRepren = torch.tanh(sequeezeR)


        y = self.hidden2label2(sentenceRepren)
        logit = y
        return logit
