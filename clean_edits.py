# clean_edits.py
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

    word_lookup = loader.make_word_lookup(names)
    print word_lookup

    correct = []
    for name in names:
        words = name.split()
        for i in range(len(words)):
            words[i] = word_lookup[loader.sort_word(words[i])][0][0]
        correct.append(' '.join(words))

    with open('corrected.txt', 'w') as outfile:
        for i in range(len(names)):
            outfile.write('%s\t%s\n' % (names[i], correct[i]))

    # for each name in order of descending frequency
    
    # Hypothesis: there is a correct version of the name somewhere in
    # the data. For each name in the dataset, especially the ones that
    # have only one or two instances, I should be able to remake the
    # name by applying the edits. For instance, one of the words is
    # probably scrambled, or maybe deleted or doubled. Maybe two of
    # the words have been swapped.

    # First, build a model that is capable of correcting the jumbled
    # word misspellings. This should be somewhat straightforward since
    # the correct words will occur with high frequency. This
    # dictionary can be indexed by the word with its letters sorted,
    # and ordered in terms of decreasing frequency. Each of the words
    # in the list for that index can then be tried in order.

    print 'all done.'
