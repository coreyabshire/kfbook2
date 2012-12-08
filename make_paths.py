# make_paths.py
#
# Implements a path file builder. The purpose is to both help me
# better understand the nature of the graph and the associated paths,
# as well as to generate additional test data.
#

import sys
import loader
import random
import networkx as nx

file_low = 1
file_high = 15
train_format = 'clean/train%d.txt'
delim = ' | '
cutoff_params = 5.6617, 1.6768

def randnode(graph):
    return graph.nodes()[random.randint(0,len(graph)-1)]

def random_shortest_path(graphs):
    mu, sigma = cutoff_params
    cutoff = int(random.gauss(mu, sigma))
    graph = graphs[random.randint(0,len(graphs)-1)]
    source = randnode(graph)
    paths = nx.single_source_shortest_path(graph, source, cutoff)
    targets = paths.keys()
    target = targets[random.randint(0,len(targets)-1)]
    path = nx.shortest_path(graph, source, target, 'weight')
    return tuple(path)

def node_presence(graphs, path):
    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: make_paths <filename> <npaths>'
        sys.exit(-1)
    filename = sys.argv[1]
    npaths = int(sys.argv[2])
    print 'loading clean training files'
    edge_lists = loader.load_train_files(train_format, file_low, file_high)
    graphs = loader.get_graphs_from_edge_lists(edge_lists)
    print 'making paths file %s' % filename
    with open(filename, 'w') as outfile:
        for i in range(npaths):
            outfile.write(delim.join(random_shortest_path(graphs)))
            outfile.write('\n')
