# names.py
#
# This module loads learns how to correct node names so they can be
# cleaned up prior to training on the network graph.

def extract_features(names):
    return []



# Unsupervised learning I don't have a dataset that says all of the
# correct node names for each node I need to process. This means I'm
# dealing with an unsupervised learning problem. The challenge is
# picking the right number of classes to use to divide the data
# into. What parameters control and how can I estimate how good it is?

# Upon training, I would like the module to be able to give me a
# unique number for each real node name.

# To do this, I need to create a data set of all the node
# names. Should they be unique? Or do I just keep the instances like
# they are?

# What should the features be? Maybe each word, individually, along
# with each word sorted. I first need to get all unique words to
# create the vocabulary file. I should keep case. I should also keep
# the punctuation and such on the words. That is, all I need to do is
# split on spaces.
