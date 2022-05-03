import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # attn_model=args.attn_model
        class_num = args.class_num
        filter_num = args.filter_num
        # seq_len=args.batch_size
        self.vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        #####################################多通道混合bert训练词向量#############################
        self.embedding = nn.Embedding(self.vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors,freeze=not args.non_static)  # non_static=FALSE 可以微调\
        # self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dimension, filter_num)

        self.fc = nn.Linear(filter_num, class_num)

    def forward(self, text):
        # text = [batch size,sent len, ]

        embedded = self.embedding(text)

        # embedded = [batch size,sent len,  emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        # assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        out=self.fc(output.sum(1))
        return out