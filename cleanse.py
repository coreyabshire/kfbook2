# cleanse.py: Command line utility to read in all the messy names from
# the competition and write out a file that makes it easy to assess
# the cleansing function.

import loader
from collections import Counter

if __name__ == '__main__':
    print 'loading data'
    paths = loader.load_paths_file('data/paths.txt')
    graphs = loader.load_all_train_files()

    print 'convert to lists from tuples for assignment'
    paths = [[x for x in path] for path in paths]
    graphs = [[[x for x in edge] for edge in graph] for graph in graphs]
    
    print 'getting all names'
    names = loader.get_all_nodes_dict(graphs, paths)

    print 'splitting off text names for corrector'
    text_names = {}
    for name, count in names.items():
        if not name.isdigit():
            text_names[name] = count

    print 'making word lookup'
    word_lookup = loader.make_word_lookup(text_names)
    
    print 'getting name corrections'
    name_found, edits = loader.get_name_found(text_names, word_lookup)

    print 'correcting graph names'
    graph_corrections = 0
    for i in range(len(graphs)):
        for j in range(len(graphs[i])):
            for k in range(2):
                name = graphs[i][j][k]
                if not name.isdigit() and name in name_found:
                    graphs[i][j][k] = name_found[name]
                    graph_corrections += 1
    print '%d corrections made' % graph_corrections

    print 'correcting path names'
    path_corrections = 0
    for i in range(len(paths)):
        path = paths[i]
        for j in range(len(path)):
            name = paths[i][j]
            if name in name_found:
                paths[i][j] = name_found[name]
                path_corrections += 1
    print '%d corrections made' % path_corrections

    print 'writing clean data'
    for i in range(len(graphs)):
        filename = 'clean/train%d.txt' % (i+1)
        graph = graphs[i]
        with open(filename, 'w') as f:
            for edge in graph:
                f.write(' | '.join([str(x) for x in edge]))
                f.write('\n')

    with open('clean/paths.txt', 'w') as f:
        for path in paths:
            f.write(' | '.join(path))
            f.write('\n')

    print 'all done.'
