
import torch.nn as nn
import torch.nn.functional as F
import torch
class CGRU(nn.Module):

    def __init__(self, args):
        super(CGRU, self).__init__()
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
        # in_channels=1
        # Co = args.filter_num
        self.embedding = nn.Embedding(V, D)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,freeze= not args.non_static)  # non_static=FALSE 可以微调\s

        # CNN
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=self.hidden_dim,
                      kernel_size=(fs, D))
            for fs in filter_sizes
        ])

        # GRU
        self.gru = nn.GRU(D, self.hidden_dim, num_layers=self.num_layers)
        # linear
        L = len(filter_sizes) * self.hidden_dim  + self.hidden_dim
        self.hidden2label1 = nn.Linear(L, L// 2)
        self.hidden2label2 = nn.Linear(L // 2, C)
        # dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embedding(x)
        # CNN
        cnn_x = embed
        # cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        # LSTM
        gru_x = embed.view(len(x), embed.size(1), -1)
        gru_out, _ = self.gru(gru_x)
        # lstm_out = torch.transpose(lstm_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        # CNN and LSTM cat
        cnn_x = torch.transpose(cnn_x, 0, 1)
        gru_out = torch.transpose(gru_out, 0, 1)
        cnn_gru_out = torch.cat((cnn_x, gru_out), 0)
        cnn_gru_out = torch.transpose(cnn_gru_out, 0, 1)

        # linear
        cnn_gru_out = self.hidden2label1(torch.tanh(cnn_gru_out))
        cnn_gru_out = self.hidden2label2(torch.tanh(cnn_gru_out))
        # output
        logit = cnn_gru_out

        return logit