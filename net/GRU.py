import torch.nn as nn
import torch.nn.functional as F
import torch
"""
Neural Networks model : GRU
"""


class GRU(nn.Module):

    def __init__(self, args):
        super(GRU, self).__init__()
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

        # gru
        self.gru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        #  dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.embedding(input)
        input = embed.view(len(input), embed.size(1), -1)
        lstm_out, _ = self.gru(input)
        # lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = torch.tanh(lstm_out)
        # linear
        y = self.hidden2label(lstm_out)
        logit = y
        return logit