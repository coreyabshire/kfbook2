# clean_short_names.py
#
# Reads in a shorter names file and just tries to clean that.
#

from collections import defaultdict
import numpy as np
import loader

if __name__ == '__main__':
    with open('names_short.txt') as infile:
        names = [name.strip() for name in infile.readlines()]
    ninstances = len(names)
    print 'read %d instances' % ninstances

    print 'extract vocabulary'
    word_counts = defaultdict(int)
    for name in names:
        words = name.split()
        for w in words:
            word_counts[w] += 1
            word_counts[loader.sort_word(w)] += 1
    vocab_words = []
    vocab_threshold = 1
    for w, c in word_counts.items():
        if c > vocab_threshold:
            vocab_words.append(w)
    vocab_words.sort()
    vocab = dict((w,i) for i,w in enumerate(vocab_words))
    nfeatures = len(vocab)
    print 'extracted %d word vocabulary' % nfeatures

    print 'extract features'
    namevecs = []
    for name in names:
        words = name.split()
        sorted_words = [loader.sort_word(w) for w in words]
        words += sorted_words
        namevec = np.zeros(nfeatures)
        for word in words:
            if word in vocab:
                namevec[vocab[word]] = 1
        namevecs.append(namevec)
    x = np.matrix(namevecs)

    sim = np.zeros((ninstances, ninstances))
    for i in range(ninstances):
        a = x[i,:]
        for j in range(ninstances):
            b = x[j,:]
            nmatches = np.logical_and(a,b).sum()
            sim[i,j] = nmatches
    np.savetxt('sim.csv', sim, delimiter=',', fmt='%d')
    
    print sim

    print 'all done.'
