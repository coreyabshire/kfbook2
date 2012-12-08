# check_name_found.py
#
# Checks the name corrector.
#
# Copyright 2012 Corey Abshire <corey.abshire@gmail.com>
#

import numpy as np
import loader
from collections import defaultdict

train_file_format = 'train/train%d.txt'
train_file_low = 1
train_file_high = 15
test_times = 5
paths_file = 'paths.txt'
submission_file = 'submission.csv'

if __name__ == '__main__':

    print 'loading data'
    paths = loader.load_paths_file(paths_file)
    graphs = loader.load_train_files(train_file_format,
                                     train_file_low, train_file_high)
    m = len(paths)
    n = test_times
    # pred = np.zeros((m,n))
    pred = np.random.rand(m,n)

    print 'training node name decoder model'
    print 'get all nodes'
    names = loader.get_all_nodes_dict(graphs, paths)
    print 'make word lookup'
    word_lookup = loader.make_word_lookup(names)
    print 'get name found'
    name_found, edits = loader.get_name_found(names, word_lookup)
    
    actual = defaultdict(int)
    corrected = defaultdict(int)
    print 'writing details'
    with open('check_name_found.txt', 'w') as outfile:
        for name in names:
            if name in name_found:
                corrected[name_found[name]] += names[name]
                outfile.write('%s\t%d\t%s\n' % (
                        name_found[name],
                        names[name],
                        name))
            else:
                actual[name] = names[name]
                outfile.write('%s\t%d\n' % (
                        name,
                        names[name]))

    print 'writing summary'
    with open('check_name_found_summary.txt', 'w') as outfile:
        for name in actual.keys():
            outfile.write('%s\t%d\t%d\t%d\n' % (
                    name,
                    actual[name],
                    corrected[name],
                    actual[name] + corrected[name]))

    print 'classifying nodes'
    print 'transforming initial graphs'
    print 'reading and cleaning data files'

    
