import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Neural Network: CNN_BiLSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""


class CNN_BiLSTM(nn.Module):

    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
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

        # Co = args.filter_num
        self.embedding = nn.Embedding(V, D)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,
                                                            freeze=not args.non_static)  # non_static=FALSE 可以微调\s


        # CNN
        # CNN
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.hidden_dim,
                      kernel_size=(fs, D))
            for fs in filter_sizes
        ])
        # print(self.convs1)
        # # for cnn cuda
        # if self.args.cuda is True:
        #     for conv in self.convs1:
        #         conv = conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)

        # linear
        L = len(filter_sizes) * self.hidden_dim  + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embedding(x)

        # CNN
        cnn_x = embed
        # cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [torch.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # BiLSTM
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        # bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = torch.tanh(bilstm_out)

        # CNN and BiLSTM CAT
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)

        # linear
        cnn_bilstm_out = self.hidden2label1(torch.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(torch.tanh(cnn_bilstm_out))

        # output
        logit = cnn_bilstm_out
        return logit