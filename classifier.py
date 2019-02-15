import torch
import torch.nn as nn
import numpy as np

import random
from collections import defaultdict
import matplotlib.pyplot as plt

import plotly
import plotly.plotly as py
import plotly.figure_factory as ff

# from cnn_model import CNN
# from cnn_pretrained import CNN
from cnn_pretrained_multichannel import CNN


MODEL_PKL = 'best_cnn_multichannel_model.pkl'
w2i = defaultdict(lambda: len(w2i))
UNK = w2i['UNK']

FILE_PREFIX = "topicclass/topicclass_"
EPOCHS = 5
BATCH_SIZE = 100
DROPOUT_P = 0.5
EMBED_DIM = 300

def load_dataset(split):

    filename = FILE_PREFIX + split + '.txt'
    f = open(filename)
    X = []
    Y = []
    for line in f:
        tag, words = line.lower().strip().split(' ||| ')
        words = words.split(" ")
        X.append([w2i[x] for x in words])
        Y.append(tag)

    return X, Y


def calc_accuracy(model, X_data, Y_data, batch_size):

    num_samples = ((len(X_data)/batch_size)*batch_size)
    correct = 0
    preds_and_tags = []
    for i in range(len(X_data)/batch_size):
        X = X_data[i*batch_size:(i+1)*batch_size]
        Y = Y_data[i*batch_size:(i+1)*batch_size]

        max_len = max([len(x) for x in X])
        batch_X = [x + [0]*(max_len-len(x)) for x in X]
        batch_X = np.array(batch_X)
        
        X_tensor = torch.tensor(batch_X)
        probs = model(X_tensor)
        predict = probs.argmax(dim=1)
        for j in range(batch_size):
            if predict[j].item() == Y[j]:
                correct += 1
            preds_and_tags.append((predict[j].item(), Y[j]))

    return correct*100.0/num_samples, preds_and_tags


def train(model, train_X, train_Y, val_X, val_Y):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    losses = []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        train_data = zip(train_X, train_Y)
        random.shuffle(train_data)

        train_X = [X for X, Y in train_data]
        train_Y = [Y for X, Y in train_data]

        iter = 0
        for i in range(len(train_X)/BATCH_SIZE):
            X = train_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            Y = train_Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

            # find maximum length sentence and pad remaining sentences
            max_len = max([len(x) for x in X])
            batch_X = [x + [0]*(max_len-len(x)) for x in X]
            batch_X = np.array(batch_X)
 
            X_tensor = torch.tensor(batch_X)
            Y_tensor = torch.tensor(Y)
            probs = model(X_tensor)
            loss = criterion(probs, Y_tensor)
            train_loss += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 10 == 0:
                losses.append(loss.item())
            if iter % 100 == 0:
                print 'Iteration', iter, '\tLoss:', loss.item()

        print 'Epoch', epoch, '\tLoss:', train_loss/(len(train_X)/BATCH_SIZE)
 
        model.eval()   
        train_acc, _ = calc_accuracy(model, train_X, train_Y, BATCH_SIZE)
        val_acc, _ = calc_accuracy(model, val_X, val_Y, 1)
        print 'Training accuracy:', train_acc
        print 'Validation accuracy:', val_acc

        if val_acc > best_val_acc:
            # torch.save(model.state_dict(), MODEL_PKL) 
            best_val_acc = val_acc

    plt.xlabel('Iterations (/10)')
    plt.ylabel('Cross Entropy Loss')
    plt.plot(range(len(losses)), losses, color='b')
    plt.legend()
    plt.show()



def tag_test_data(best_model, X_data, batch_size):

    preds = []
    for i in range(len(X_data)/batch_size):
        X = X_data[i*batch_size:(i+1)*batch_size]

        max_len = max([len(x) for x in X])
        batch_X = [x + [0]*(max_len-len(x)) for x in X]
        batch_X = np.array(batch_X)
        
        X_tensor = torch.tensor(batch_X)
        probs = best_model(X_tensor)
        predict = probs.argmax(dim=1)
        for j in range(batch_size):
            preds.append(predict[j].item())

    return preds

def write_test_tags(best_model, X_data, tags):

    preds = tag_test_data(best_model, X_data, 1)
    preds = [tags[t] for t in preds]
    print 'Number of predictions:', len(preds)
    preds_file = open('test_tags.txt', 'w')
    for j in preds:
        preds_file.write(str(j)+'\n')
    preds_file.close()
                   


def confusion_matrix(preds_and_tags, tags):

    count = {}
    for t1 in range(ntags):
        for t2 in range(ntags):
            count[(t1, t2)] = 0
    for pt in preds_and_tags:
        count[pt] += 1

    conf = []
    for i in range(ntags)[::-1]:
        row = [count[(i,j)] for j in range(ntags)]
        conf.append(row)
    fig = ff.create_annotated_heatmap(conf, x=tags, y=tags[::-1], colorscale='Viridis')
    plotly.offline.plot(fig, filename='conf_matrix_heatmap')
    

train_X, train_Y = load_dataset('train')
# train_X = train_X[:1000]
# train_Y = train_Y[:1000]
nwords = len(w2i)
# print len(w2i)
w2i = defaultdict(lambda: UNK, w2i)


val_X, val_Y = load_dataset('valid')
test_X, _ = load_dataset('test')

print 'Number of training instances:', len(train_X)
print 'Number of validation instances:', len(val_X)
print 'Number of testing instances:', len(test_X)

# Convert string tags to indices
tags = []
for tag in train_Y:
    if tag not in tags:
        tags.append(tag)
train_Y = [tags.index(t) for t in train_Y]
val_Y = [tags.index(t) for t in val_Y]

tag_abbrs = ['SS', 'SR', 'NS', 'LL', 'GP', 'Mu', 'MD', 'AA', 'W', 'ET', 'VG', 'PR', 'AFD', 'H', 'Ma', 'Misc']

# nwords = len(w2i)
print 'nwords:', nwords
ntags = len(tags)
print 'ntags:', ntags

num_filters = [100, 100, 100]
kernel_sizes = [3, 4, 5]

model = CNN(nwords, EMBED_DIM, num_filters, kernel_sizes, ntags, w2i, DROPOUT_P)

train(model, train_X, train_Y, val_X, val_Y)


best_model = CNN(nwords, EMBED_DIM, [100, 100, 100], [3, 4, 5], ntags, w2i, DROPOUT_P)
best_model.load_state_dict(torch.load(MODEL_PKL))
best_model.eval()

# write_test_tags(best_model, test_X, tags)

acc, preds_and_tags = calc_accuracy(best_model, val_X, val_Y, 1)
print 'Validation accuracy:', acc

confusion_matrix(preds_and_tags, tag_abbrs)
