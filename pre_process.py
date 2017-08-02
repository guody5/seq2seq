from sklearn import  preprocessing
import pandas as pd
import numpy as np
class Data(object):
    def __init__(self):
        self.index=0


    def load_data(self):
        #This function to load data and word encode to number
        train_x=[]
        train_y=[]
        valid_x=[]
        valid_y=[]
        test_x=[]
        test_y=[]
        with open('data/train.query.tok','r') as f:
            for line in f:
                train_x.append(line[:-1].split(' '))
        with open('data/train.reply.tok','r') as f:
            for line in f:
                train_y.append(line[:-1].split(' '))
        with open('data/test.query.tok','r') as f:
            for line in f:
                test_x.append(line[:-1].split(' '))
        with open('data/test.reply.tok','r') as f:
            for line in f:
                test_y.append(line[:-1].split(' '))
        with open('data/valid.query.tok','r') as f:
            for line in f:
                valid_x.append(line[:-1].split(' '))
        with open('data/valid.reply.tok','r') as f:
            for line in f:
                valid_y.append(line[:-1].split(' '))
        vocabulary=[]
        for line in train_x+train_y+test_x+test_y+valid_x+valid_y:
            vocabulary+=line
        dict={}
        for word in vocabulary:
            try:
                dict[word]+=1
            except:
                dict[word]=1
        index=2
        cont_vocabulary=sorted(dict.items(), key=lambda d: d[1],reverse=True)
        for word,cont in cont_vocabulary:
            dict[word]=index
            index+=1
        self.train_x=list(map(lambda x:[dict[word] for word in x],train_x))
        self.train_y = list(map(lambda x: [dict[word] for word in x]+[1], train_y))
        self.valid_x = list(map(lambda x: [dict[word] for word in x], valid_x))
        self.valid_y = list(map(lambda x: [dict[word] for word in x]+[1], valid_y))
        self.test_x = list(map(lambda x: [dict[word] for word in x], test_x))
        self.test_y = list(map(lambda x: [dict[word] for word in x]+[1], test_y))
        self.dict=dict

    def vocab_size(self):
        return len(dict)

    def pad_sequences(self,sequences, maxlen=None, dtype='int32',
                      padding='post', truncating='post', value=0.):
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return  x

    def next_batch(self,batch_size,max_len=None):
        #input: batch size
        #output: batch training data,epoch
        if max_len==None:
            batch_padding=True
        else:
            batch_padding=False
        new_index=min(len(self.train_x),self.index+batch_size)
        batch_x=self.train_x[self.index:new_index]
        batch_y=self.train_y[self.index:new_index]
        self.index=new_index
        if self.index==len(self.train_x):
            epoch=True
            self.index=0
        else:
            epoch=False
        #padding
        if batch_padding:
            batch_x=self.pad_sequences(batch_x)
            batch_y=self.pad_sequences(batch_y)
        else:
            batch_x=self.pad_sequences(batch_x,max_len)
            batch_y=self.pad_sequences(batch_y,max_len)

        return batch_x,batch_y,epoch



