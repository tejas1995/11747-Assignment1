import torch
import torch.nn as nn
import numpy as np

import gensim
import sys


class CNN(nn.Module):

    def __init__(self, nwords, embed_dim, num_filters, kernel_sizes, num_classes, w2i, drop_p, fixed_embeds=False):

        super(CNN, self).__init__()

        # load word2vec weights for our w2i vocabulary
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        vocab_weights = {}
        for w, i in w2i.iteritems():
            if w in word2vec_model.wv.vocab:
                vocab_weights[i] = word2vec_model.get_vector(w)
            else:
                vocab_weights[i] = word2vec_model.get_vector("UNK")
        print 'Loaded word2vec weights...'
        pt_weights = [vocab_weights[i] for i in range(nwords)]
        pt_weights = np.array(pt_weights)
        pt_weights = torch.FloatTensor(pt_weights)
        
        self.embed = nn.Embedding.from_pretrained(pt_weights, freeze=fixed_embeds)

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

        emb = self.embed(sentence)      # 300 x nwords
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

