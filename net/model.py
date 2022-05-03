import torch
import torch.nn as nn
import torch.nn.functional as F
class attenTextCNN(nn.Module):
    def __init__(self, args):
        super(attenTextCNN, self).__init__()
        self.args = args
        chanel_num = 1
        self.vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        #####################################多通道混合bert训练词向量#############################
        self.embedding = nn.Embedding(self.vocabulary_size, embedding_dimension)
        self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        self.embedding.weight.data.copy_(args.vectors)
        # self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.static:
            # non_static=FALSE 可以微调
            chanel_num += 1
        if args.multichannel:
            chanel_num += 1
        else:
            self.embedding2 = None
        filter_num = args.filter_num
        filters_size = args.filter_sizes
        #############################注意力机制##################################################

        self.global_att = GlobalAttention(args, embedding_dimension)
        self.lobal_att = LocalAttention(args, embedding_dimension)
        self.n = 0.5
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=2,
                      out_channels=filter_num,
                      kernel_size=(fs, embedding_dimension), padding=((fs-1)//2, 0))
            if fs % 2 != 0 else nn.Conv2d(in_channels=2,
                                          out_channels=filter_num,
                                          kernel_size=(fs, embedding_dimension), padding=((fs - 2) // 2, 0))
             for fs in filters_size
        ])
        args.chanel_num=chanel_num
        self.drop=nn.Dropout(args.dropout)

        self.share = nn.Conv2d(1, 1, kernel_size=1)
        self.share1 = nn.Conv2d(1, 1, kernel_size=1)
        # self.fc = nn.Linear(embedding_dimension, embedding_dimension)
        self.hiddens = nn.Linear(embedding_dimension, args.class_num)

        #################################初始化##################################################
        self.reset_para()
    def forward(self, x):

        embed=self.embedding(x)

        out1 = self.lobal_att(embed)+self.share( embed.unsqueeze(1)).squeeze(1)
        # lobal_att  对混合随机词向量+预训练词向量求导 卷积三次   注意

        out2 = self.global_att(embed)+self.share1( embed.unsqueeze(1)).squeeze(1)
        # global_att  对混合随机词向量+预训练词向量求导 卷积三次   注意F.tanh()
        out=torch.tanh(torch.stack((out1,out2),dim=1))

        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        # torch.save({'global': out2.squeeze(0), 'local': out1.squeeze(0), 'conv':[i.squeeze(0) for i in conved]}, '../x.pt')
        out = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.drop(torch.cat(out, dim=1))
        logits=self.hiddens(cat)
        logits=torch.softmax(logits,dim=1)
        return logits
    def reset_para(self):
        nn.init.xavier_normal_(self.lobal_att.att_conv[0].weight, gain=1)
        # nn.init.constant_(self.lobal_att.att_conv[0].bias, 0.1)
        for liner in [self.global_att.proj_w,self.hiddens,self.global_att.proj_v]:
            nn.init.uniform_(liner.weight, -0.1, 0.1)
            # nn.init.constant_(liner.bias, 0.1)
        # nn.init.normal_(self.hiddens2.weight,-0.1, 0.1)
        nn.init.xavier_normal_(self.share.weight, gain=1)
        nn.init.xavier_normal_(self.share1.weight, gain=1)
        for cnn in self.convs:
            nn.init.kaiming_normal_(cnn.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            # nn.init.constant_(cnn.bias, 0.1)

class LocalAttention(nn.Module):
    def __init__(self, args, embedding_dimension):
        super(LocalAttention, self).__init__()
        self.args = args

        self.att_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(5,embedding_dimension), padding=(2, 0)),

            nn.Sigmoid()
        )
    def forward(self, x):
        # x=torch.bmm(x.transpose(1,2),x)
        score = self.att_conv(x.unsqueeze(1)).squeeze(1)
        out = x.mul(score)
        return out

# #
class GlobalAttention(nn.Module):
    def __init__(self, args, embedding_dimension):
        super(GlobalAttention, self).__init__()
        self.args = args
        self.proj_w = nn.Linear(embedding_dimension, args.filter_num, bias=0)
        self.proj_v = nn.Linear(args.filter_num, 1, bias=1)
    def forward(self, x):
        # x=torch.bmm(x.transpose(1,2),x)
        # s(x, q) = v.T * sigmoid (W * x + b)
        mlp_x = self.proj_w(x)
        att_scores = torch.sigmoid(self.proj_v(F.relu(mlp_x)))
        out = x*att_scores
        return out
