import torch
import torch.nn as nn
import numpy as np

import gensim
import sys


class CNN(nn.Module):

    def __init__(self, nwords, embed_dim, num_filters, kernel_sizes, num_classes, w2i, drop_p):

        super(CNN, self).__init__()

        self.embed = nn.Embedding(nwords, embed_dim)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)

        conv_1d_list = []
        for num_filter, kernel_size in zip(num_filters, kernel_sizes):
            conv_1d_list.append(nn.Conv1d(embed_dim, num_filter, kernel_size))
        self.conv_1ds = nn.ModuleList(conv_1d_list)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_p)

        self.total_filters = sum(num_filters)
        self.fully_connected = nn.Linear(self.total_filters, num_classes)
        nn.init.xavier_uniform_(self.fully_connected.weight)
        


    def forward(self, sentence):

        emb = self.embed(sentence)     
        emb = emb.permute(0, 2, 1)
        ftrs = []
        for conv_1d in self.conv_1ds:
            ftr = conv_1d(emb)
            pooled = ftr.max(dim=2)[0]
            ftrs.append(pooled)

        cat_ftrs = torch.cat(ftrs, dim=1)
        h = self.activation(cat_ftrs)
        h = self.dropout(h)
        out = self.fully_connected(h)

        return out

