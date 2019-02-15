import torch
import torch.nn as nn
import numpy as np

import gensim
import sys


class CNN(nn.Module):

    def __init__(self, nwords, embed_dim, num_filters, kernel_sizes, num_classes, w2i, drop_p):

        super(CNN, self).__init__()

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
        
        self.fixed_embed = nn.Embedding.from_pretrained(pt_weights, freeze=True)
        self.tuned_embed = nn.Embedding.from_pretrained(pt_weights, freeze=False)

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

        fixed_emb = self.fixed_embed(sentence)      # 300 x nwords
        fixed_emb = fixed_emb.permute(0, 2, 1)

        tuned_emb = self.tuned_embed(sentence)      # 300 x nwords
        tuned_emb = tuned_emb.permute(0, 2, 1)
        ftrs = []
        for conv_1d in self.conv_1ds:
            fixed_ftr = conv_1d(fixed_emb)
            tuned_ftr = conv_1d(tuned_emb)
            comb_ftr = torch.add(fixed_ftr, tuned_ftr)
            pooled = comb_ftr.max(dim=2)[0]
            ftrs.append(pooled)

        cat_ftrs = torch.cat(ftrs, dim=1)
        h = self.activation(cat_ftrs)
        h = self.dropout(h)
        out = self.fully_connected(h)

        return out

