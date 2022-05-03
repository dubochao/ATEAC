import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Description:
    the model is a mulit-channel CNNS model, 
    the model use two external word embedding, and then,
    one of word embedding built from train/dev/test dataset,
    and it be used to no-fine-tune,other one built from only
    train dataset,and be used to fine-tune.

    my idea,even if the word embedding built from train/dev/test dataset, 
    whether can use fine-tune, in others words, whether can fine-tune with
    two external word embedding.
"""


class CNN_MUI(nn.Module):

    def __init__(self, args):
        super(CNN_MUI, self).__init__()
        self.args = args
        # attn_model=args.attn_model
        # class_num = args.class_num
        # filter_num = args.filter_num
        # seq_len=args.batch_size
        filter_sizes = args.filter_sizes

        self.hidden_dim = args.filter_num
        self.num_layers = args.layers
        V = args.vocabulary_size
        D = args.embedding_dim
        C = args.class_num
        in_channels=1
        # Co = args.filter_num
        self.embedding = nn.Embedding(V, D)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,freeze= args.non_static)  # non_static=FALSE 可以微调\s
        if args.multichannel:
            self.embedding2 = nn.Embedding(V, D).from_pretrained(args.vectors)
            in_channels+=1

        # CNN
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.hidden_dim,
                      kernel_size=(fs, D))
            for fs in filter_sizes
        ])

        # if args.init_weight:
        #     print("Initing W .......")
        #     for conv in self.convs:
        #         nn.init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
        #         nn.init.uniform(conv.bias, 0, 0)
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)

        # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()
        in_fea = len(filter_sizes) * self.hidden_dim
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)

        # if args.batch_normalizations is True:
        print("using batch_normalizations in the model......")
        self.convs1_bn = nn.BatchNorm2d(num_features=self.hidden_dim, eps=1e-05, momentum=0.1, affine=True)
        self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=0.1, affine=True)
        self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=0.1, affine=True)
    #
    # def conv_and_pool(self, x, conv):
    #     x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
    #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
    #     return x

    def forward(self, x):
        x_no_static = self.embedding(x)
        x_static = self.embedding2(x)
        x = torch.stack([x_static, x_no_static], 1)
        x = self.dropout(x)

        x = [F.relu(self.convs1_bn(conv(x))).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # else:
        #     x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        #     x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)

        x = self.fc1(x)
        x = self.fc1_bn(x)
        logit = self.fc2(F.relu(x))
        logit = self.fc2_bn(logit)
        return logit