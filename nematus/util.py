'''
Utility functions
'''

import cPickle as pkl
import exception
import json
import numpy
import sys
import math

# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# batch preparation
def prepare_data(seqs_x, seqs_y, n_factors, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seq2words(seq, inverse_dictionary, join=True):
    seq = numpy.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_dictionary],
                             join)

def factoredseq2words(seq, inverse_dictionaries, join=True):
    if type(seq)==list:
        seq=numpy.array(seq)
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    for i, w in enumerate(seq):
        factors = []
        for j, f in enumerate(w):
            if f == 0:
                assert (i == len(seq) - 1) or (seq[i+1][j] == 0), \
                       ('Zero not at the end of sequence', seq)
            elif f in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][f])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words

def reverse_dict(dictt):
    keys, values = zip(*dictt.items())
    r_dictt = dict(zip(values, keys))
    return r_dictt


def load_dictionaries(config):
    source_to_num = [load_dict(d) for d in config.source_dicts]
    target_to_num = load_dict(config.target_dict)
    num_to_source = [reverse_dict(d) for d in source_to_num]
    num_to_target = reverse_dict(target_to_num)
    return source_to_num, target_to_num, num_to_source, num_to_target


def read_all_lines(config, sentences, batch_size):
    source_to_num, _, _, _ = load_dictionaries(config)

    if config.source_vocab_sizes != None:
        assert len(config.source_vocab_sizes) == len(source_to_num)
        for d, vocab_size in zip(source_to_num, config.source_vocab_sizes):
            if vocab_size != None and vocab_size > 0:
                for key, idx in d.items():
                    if idx >= vocab_size:
                        del d[key]

    lines = []
    for sent in sentences:
        line = []
        for w in sent.strip().split():
            if config.factors == 1:
                w = [source_to_num[0][w] if w in source_to_num[0] else 1]
            else:
                w = [source_to_num[i][f] if f in source_to_num[i] else 1
                                         for (i,f) in enumerate(w.split('|'))]
                if len(w) != config.factors:
                    raise exception.Error(
                        'Expected {0} factors, but input word has {1}\n'.format(
                            config.factors, len(w)))
            line.append(w)
        lines.append(line)
    lines = numpy.array(lines)
    lengths = numpy.array(map(lambda l: len(l), lines))
    lengths = numpy.array(lengths)
    idxs = lengths.argsort()
    lines = lines[idxs]

    #merge into batches
    batches = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batches.append(batch)

    return batches, idxs

def squeeze(listt): # [[1,2],[3,4],[5,6]] => [1,2,3,4,5,6]
    result = []
    for i in listt:
        for j in i :
            result.append(j)
    return result

def wrap_elements(listt):
    return [[ele] for ele in listt]

def unwrap_elements(listt):
    if len(listt)==1:
        return listt[0]
    ans = []
    for i in listt:
        ans.append(unwrap_elements(i))
    return ans

def mean(listt):
    s = 0.
    l =  len(listt)
    for i in listt:
        if type(i)==list:
            print "warning: list received",i
            s += i[0]
        else:
            s += i
    return s/l

def std(listt):
    s = 0.
    m = mean(listt)
    l =  len(listt)
    for i in listt:
        if type(i)==list:
            print "warning: list received",i
            s += (i[0]-m)**2
        else:
            s += (i-m)**2
    return (s/l)**(0.5)

def standardlize(listt, m=None,s=None):
    if m == None:
        m = mean(listt)
    if s == None:
        s = std(listt)
    if (s == 0):
        print ('WARNING, std == 0')
        return [1]*len(listt)
    return [ ( (x[0] - m) / s if type(x)==list else ((x - m) / s)) for x in listt] 

def update_mean(old_m=0, old_n=0, new_val=0):
    if old_n==0:
        return new_val
    return (new_val+old_n*old_m)/(old_n+1)

def update_std(old_std=0, old_m=0, old_n=0,new_val=0):
    if old_n==0:
        return 0
    old_std2 = old_std**2
    new_n = old_n+1
    new_std2 = (old_n*(new_val-old_m)**2/(new_n**2))+(old_n*old_std2)/new_n

    return new_std2**0.5

def update_num(n=0, add=1):
    return n+add

def softmax(l):
    el = [math.exp(i) for i in l]
    Z = sum(el)
    return [i/Z for i in el]

def split_list(l,s):
    return [l[i:i+s] for i in range(0,len(l),s)]