import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        # attn_model=args.attn_model
        class_num = args.class_num
        filter_num = args.filter_num
        # seq_len=args.batch_size
        filter_sizes = args.filter_sizes
        self.vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        #####################################多通道混合bert训练词向量#############################
        self.embedding = nn.Embedding(self.vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,
                                                            freeze=not args.non_static)  # non_static=FALSE 可以微调\
        # self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=filter_num,
                      kernel_size=(fs, embedding_dimension))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, text):


        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]
        out =self.fc(cat)

        return out