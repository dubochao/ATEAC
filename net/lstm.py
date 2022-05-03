
import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
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

        # lstm
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)

        # if args.init_weight:
        # print("Initing W .......")
        # n = self.lstm.input_size * self.lstm
        nn.init.xavier_normal_(self.lstm.all_weights[0][0], gain=1)
        nn.init.xavier_normal_(self.lstm.all_weights[0][1], gain=1)

        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # self.dropout_embed = nn.Dropout(args.dropout_embed)

    def forward(self, x):
        embed = self.embedding(x)
        embed = self.dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)
        # lstm
        lstm_out, _ = self.lstm(x)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        # lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        # lstm_out = torch.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = torch.tanh(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit