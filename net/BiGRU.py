
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):

    def __init__(self, args):
        super(BiGRU, self).__init__()
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
        # gru
        self.bigru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)
        # gru
        gru_out, _ = self.bigru(x)

        # gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        # gru_out = F.tanh(gru_out)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.tanh(gru_out)
        # linear
        y = self.hidden2label(gru_out)
        logit = y
        return logit
