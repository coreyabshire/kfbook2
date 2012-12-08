# loader.py
#
# This is the loader module for the kfbook2 entry.  This module is
# only responsible for loading the various datasets in from disk.

from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import numpy as np

def load_train_file(filename):
    lines = [s.strip() for s in file(filename).readlines()]
    graph = [tuple(s.strip() for s in a.split('|')) for a in lines]
    graph = [(a, b, int(c)) for a,b,c in graph]
    return graph

def load_train_files(format, low, high):
    graphs = []
    for i in range(low, high + 1):
        filename = format % i
        graphs.append(load_train_file(filename))
    return graphs

def load_all_train_files():
    return load_train_files('data/train%d.txt', 1, 15)

def load_paths_file(filename):
    lines = [s.strip() for s in file(filename).readlines()]
    paths = [tuple(s.strip() for s in a.split('|')) for a in lines]
    return paths

def get_names_list(graphs, paths):
    names = []
    for graph in graphs:
        for name1, name2, cost in graph:
            names.append(name1)
            names.append(name2)
    for path in paths:
        for name in path:
            names.append(name)
    return names

def split_names_list(all_names):
    numeric = []
    alphanumeric = []
    for name in all_names:
        if name.isdigit():
            numeric.append(name)
        else:
            alphanumeric.append(name)
    return numeric, alphanumeric

def get_word_counts(names):
    counts = {}
    for name in names:
        for word in name.split():
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts

def get_names_vocab(word_counts, cutoff=2):
    vocab = []
    for word, count in word_counts.items():
        if count > cutoff:
            vocab.append(word)
    return vocab

def get_nodes_dict(graph):
    nodes = defaultdict(int)
    for a, b, c in graph:
        nodes[a] += 1
        nodes[b] += 1
    return nodes

def get_all_nodes_dict(graphs, paths):
    nodes = defaultdict(int)
    for graph in graphs:
        for a, b, c in graph:
            nodes[a] += 1
            nodes[b] += 1
    for path in paths:
        for x in path:
            nodes[x] += 1
    return nodes

def fix_names(graph):
    # get all the names in the graph
    # split each name into its component words
    return graph

def sort_word(w):
    return ''.join(sorted(w))

def make_index(word_counts):
    index = defaultdict(list)
    for w,c in word_counts.items():
        index[sort_word(w)].append((w,c))
    return index

def make_word_lookup(names):
    word_counts = defaultdict(lambda: defaultdict(int))
    for name in names:
        for word in name.split():
            key = sort_word(word)
            word_counts[key][word] += 1
    word_lookup = {}
    for key, counts in word_counts.items():
        word_lookup[key] = sorted(counts.items(), key=lambda x: x[1],
                                  reverse=True)
    return word_lookup

def get_name_found(names, word_lookup):
    name_found = {}
    edits = defaultdict(list)
    ordered_names = sorted(names.items(), key=lambda x: x[1],
                           reverse=True)
    for name, count in ordered_names:
        if name in name_found:
            continue
        words = name.split()
        for i in range(len(words)):
            k = sort_word(words[i])
            for w, c in word_lookup[k]:
                if w != words[i]:
                    new_words = words[:i] + [w] + words[i+1:]
                    new_name = ' '.join(new_words)
                    if new_name in names and new_name not in name_found:
                        name_found[new_name] = name
                        edits[name].append(('S', new_name, names[name],
                                            names[new_name]))
        words = name.split()
        for i in range(len(words)):
            for j in range(len(words)):
                new_words = words[:j] + [words[i]] + words[j:]
                new_name = ' '.join(new_words)
                if new_name in names and new_name not in name_found:
                    name_found[new_name] = name
                    edits[name].append(('I', new_name, names[name],
                                        names[new_name]))
        words = name.split()
        for i in range(len(words)):
            new_words = words[:i] + words[i+1:]
            new_name = ' '.join(new_words)
            if new_name in names and new_name not in name_found:
                name_found[new_name] = name
                edits[name].append(('D', new_name, names[name],
                                    names[new_name]))
    return name_found, edits

def get_name_found_old(names, word_lookup):
    name_found = {}
    ordered_names = sorted(names.items(), key=lambda x: x[1], reverse=True)
    for name, count in ordered_names:
        if name in name_found:
            continue
        words = name.split()
        for i in range(len(words)):
            for j in range(len(words)):
                new_words = words[:j] + [words[i]] + words[j:]
                new_name = ' '.join(new_words)
                if new_name in names and new_name not in name_found:
                    name_found[new_name] = name
        for i in range(len(words)):
            new_words = words[:i] + words[i+1:]
            new_name = ' '.join(new_words)
            if new_name in names and new_name not in name_found:
                name_found[new_name] = name
        for i in range(len(words)):
            k = sort_word(words[i])
            for w, c in word_lookup[k]:
                if w != words[i]:
                    new_words = words[:i] + [w] + words[i+1:]
                    new_name = ' '.join(new_words)
                    if new_name in names and new_name not in name_found:
                        name_found[new_name] = name
    return name_found

def get_graphs_from_edge_lists(edge_lists):
    graphs = [nx.DiGraph() for i in range(len(edge_lists))]
    for i in range(len(edge_lists)):
        graphs[i].add_weighted_edges_from(edge_lists[i])
    return graphs



def compute_cost(graph, path):
    '''
    Compute the cost of PATH in GRAPH. This is used to determine
    whether a path given in the dataset was the shortest in this
    particular graph. This cost is compared to the cost found by
    actually searching for the shortest path.
    '''
    if len(path) == 1:
        # Some of the paths are single nodes, which only go to
        # themselves. As per the FAQ, these paths are to be simply
        # treated as cost 0, and are thus by definition an optimal
        # path. (but does this include if the node no longer exists?)
        return 0
    else:
        # Walk through the path a node at a time to sum up the given
        # path. If at any point in the path a node or edge no longer
        # exists, then the path cost is undefined and we return None.
        cost = 0
        node = path[0]
        for i in range(len(path) - 1):
            if path[i] in graph:
                if path[i + 1] in graph[path[i]]:
                    cost += graph[path[i]][path[i + 1]]['weight']
                else:
                    return None
            else:
                return None
        return cost

def write_path_costs(graphs, paths, filename):
    '''
    Write out a file containing the computed cost for each path in the
    dataset across all graphs in the dataset.
    '''
    with open(filename, 'w') as f:
        for path in paths:
            src = path[0]
            dst = path[-1]
            f.write('%s\t' % src)
            f.write('%s\t' % dst)
            for i in range(len(graphs)):
                cost = compute_cost(graphs[i], path)
                if cost:
                    f.write('%d\t' % cost)
                else:
                    f.write('\t')
            f.write('\n')


def compute_path_costs(graphs, paths):
    '''
    Calculate the computed cost for each path in the dataset across
    all graphs in the dataset.
    '''
    costs = []
    for graph in graphs:
        costs.append([compute_cost(graph, path) for path in paths])
    return costs

def write_path_costs2(graphs, paths, filename):
    '''
    Write out a file to use for visually comparing the costs
    determined by checking the paths given in the dataset to the costs
    found by running a shortest path algorithm on the graph using the
    same source and target as specified in the path.
    '''
    with open(filename, 'w') as f:
        for pi in range(len(paths)):
            path = paths[pi]
            src = path[0]
            dst = path[-1]
            f.write('%s\t' % src)
            f.write('%s\t' % dst)
            f.write('|\t') # to make excel not leak into data cols
            print '%6d' % pi,
            for i in range(len(graphs)):
                cost = compute_cost(graphs[i], path)
                if cost:
                    f.write('%d\t' % cost)
                else:
                    f.write('\t')
            for i in range(len(graphs)):
                try:
                    s_cost = nx.shortest_path_length(graphs[i], src,
                                                     dst, weight='weight')
                    f.write('%d\t' % s_cost)
                    print s_cost,
                except nx.NetworkXError as e:
                    print 'x',
                    f.write('\t')
                except nx.NetworkXNoPath as e:
                    print 'y',
                    f.write('\t')
                except KeyError as e:
                    print 'z',
                    f.write('\t')
            f.write('\n')
            print

def write_shortest_path_costs(graph, paths, filename):
    '''
    Writes out a file that has one line per path, each of which is the
    shortest path found from the beginning to the end of the path in
    the graph.
    '''
    with open(filename, 'w') as f:
        for pi in range(len(paths)):
            path = paths[pi]
            src = path[0]
            dst = path[-1]
            try:
                s_cost = nx.shortest_path_length(graph, src, dst,
                                                 weight='weight')
                f.write('%d\n' % s_cost)
            except nx.NetworkXError as e:
                f.write('-1\n')
            except nx.NetworkXNoPath as e:
                f.write('-2\n')
            except KeyError as e:
                f.write('-3\n')

def load_shortest_path_costs(pattern, low, high):
    '''
    Load up a previously calculated set of shortest paths for each of
    the source and destinations specified in the paths file. Return a
    list of lists, with the outer list representing the graphs
    measured against and the inner list representing the cost for each
    path source and target.
    '''
    costs = []
    def transform(s):
        cost = int(s)
        if cost >= 0:
            return cost
        else:
            return None
    for i in range(low, high + 1):
        filename = pattern % i
        with open(filename) as f:
            costs.append([transform(s) for s in f.readlines()])
    return costs
        
def get_node_presence_matrix(graphs, paths):
    '''
    The probability of a path is likely tied to the probability of a
    node being up (among other things). We can visualize this by
    capturing 1 or 0 based on presence of a node in a graph, across
    all graphs. We can then load this into excel and color code it to
    look for patterns. This may also be helpful in pinpointing
    remaining name quality issues.
    '''
    ngraphs = len(graphs)
    ngi = {} # node graph indices
    for i in range(ngraphs):
        for node in graphs[i].nodes():
            if node not in ngi:
                ngi[node] = []
    for path in paths:
        for node in path:
            if node not in ngi:
                ngi[node] = []
    for i in range(ngraphs):
        for node in graphs[i].nodes():
            ngi[node].append(i)
    nodes = sorted(ngi.keys())
    nnodes = len(nodes)
    npm = np.zeros((nnodes, ngraphs))
    for i in range(nnodes):
        npm[i,:] = [j in ngi[nodes[i]] and 1 or 0 for j in range(ngraphs)]
    return npm, nodes

def get_edge_presence_matrix(graphs, paths):
    '''
    Like with the node presence, edge stability over time has an
    impact on path stability. Edge stability will be somewhat
    dependent on node stability, as an edge cannot be present in a
    graph if either of the related nodes are not present. However, the
    inverse is not true. Two nodes can be present and the node still
    not be there.
    '''
    ngraphs = len(graphs)
    egi = {} # edge graph indices
    for i in range(ngraphs):
        for edge in graphs[i].edges():
            if edge not in egi:
                egi[edge] = []
    for path in paths:
        for i in range(len(path) - 1):
            if edge not in egi:
                edge[(path[i], path[i+1])] = []
    for i in range(ngraphs):
        for edge in graphs[i].edges():
            egi[edge].append(i)
    edges = sorted(egi.keys())
    nedges = len(edges)
    epm = np.zeros((nedges, ngraphs))
    for i in range(nedges):
        epm[i,:] = [j in egi[edges[i]] and 1 or 0 for j in range(ngraphs)]
    return epm, edges

def get_edge_cost_matrix(graphs, paths):
    '''
    Another item that could cause optimal path changes is if the
    weight of an edge changes over time. This function returns a
    matrix over all edges as rows, with a column per graph, where each
    cell is the weight of that edge, or -1 if the edge was not found
    in the graph.
    '''
    ngraphs = len(graphs)
    egi = {} # edge graph indices
    for i in range(ngraphs):
        for edge in graphs[i].edges():
            if edge not in egi:
                egi[edge] = []
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if edge not in egi:
                egi[edge] = []
    for i in range(ngraphs):
        for edge in graphs[i].edges():
            egi[edge].append(i)
    edges = sorted(egi.keys())
    nedges = len(edges)
    ecm = np.zeros((nedges, ngraphs))
    for i in range(nedges):
        source, target = edges[i]
        for j in range(ngraphs):
            if j in egi[edges[i]]:
                ecm[i,j] = graphs[j][source][target]['weight']
            else:
                ecm[i,j] = -1
    return ecm, edges

def get_shortest_path_matrix(graphs, paths, scosts, pcosts):
    '''
    Does the same idea as for the nodes and edge presence matrices but
    for the overall shortest path. This could be a simplified version
    of the X upon which to run logistic regression.
    '''
    data = np.zeros((len(paths), len(graphs)))
    for j in range(len(graphs)):
        for i in range(len(paths)):
            if pcosts[j][i]:
                if scosts[j][i]:
                    if pcosts[j][i] <= scosts[j][i]:
                        data[i][j] = 1
                    else:
                        data[i][j] = 0
                else:
                    data[i][j] = 1
            else:
                data[i][j] = 0

    return data

def plot_roc_curve(ytest, yprob):
    '''
    This was an ROC curve implementation in use before adoption of the
    ones built-in to sklearn.
    '''
    n = 100
    pvec = np.zeros(n)
    rvec = np.zeros(n)
    fvec = np.zeros(n)
    tvec = np.zeros(n)
    for i in range(n):
        threshold = float(i) / float(n)
        ypred = yprob > threshold
        tp = sum(np.logical_and(ytest == 1, ypred == 1))
        fp = sum(np.logical_and(ytest == 0, ypred == 1))
        tn = sum(np.logical_and(ytest == 0, ypred == 0))
        fn = sum(np.logical_and(ytest == 1, ypred == 0))
        p = float(tp) / (tp + fp)
        r = float(tp) / (tp + fn)
        pvec[i] = p
        rvec[i] = r
        fpr = float(fp) / (fp + tn)
        tpr = float(tp) / (tp + fn)
        fvec[i] = fpr
        tvec[i] = tpr
        print tp, fp, tn, fn, p, r, fpr, tpr
    return fvec, tvec

def compute_accuracy(ytest, ypred):
    '''
    Computes the accuracy statistic of the given prediction set ypred
    by comparing it to the true y values in ytest.
    '''
    correct = sum(ytest == ypred)
    accuracy = float(correct) / float(len(ytest))
    return accuracy

def simple_logistic_regression(graphs, paths, scosts, pcosts):
    '''
    A little routine to capture how I've been running logistic
    regression against the dataset and compute the accuracy of it.
    '''
    data = get_shortest_path_matrix(graphs, paths, scosts, pcosts)
    x = data[:,0:10]
    y = data[:,10]
    xtest = data[:,2:12]
    ytest = data[:,12]
    #lm = linear_model.LogisticRegression()
    lm = RandomForestClassifier()
    lm.fit(x, y)
    ypred = lm.predict(xtest)
    yprob = lm.predict_proba(xtest)[:,1]
    return ytest, ypred, yprob

def logistic_regression(data):
    '''
    A little routine to capture how I've been running logistic
    regression against the dataset and compute the accuracy of it.
    '''
    x = data[:,0:13]
    y = data[:,13]
    xtest = data[:,1:14]
    ytest = data[:,14]
    lm = LogisticRegression()
    lm.fit(x, y)
    ypred = lm.predict(xtest)
    yprob = lm.predict_proba(xtest)[:,1]
    return ytest, ypred, yprob

def logistic_regression_test(data, test, ycol):
    '''
    A little routine to capture how I've been running logistic
    regression against the dataset and compute the accuracy of it.
    '''
    x = data[:,0:ycol]
    y = data[:,ycol]
    xtest = test[:,0:ycol]
    ytest = test[:,ycol]
    lm = LogisticRegression()
    lm.fit(x, y)
    ypred = lm.predict(x)
    yprob = lm.predict_proba(x)[:,1]
    ytestpred = lm.predict(xtest)
    ytestprob = lm.predict_proba(xtest)[:,1]
    return y, ypred, yprob, ytest, ytestpred, ytestprob

def run_logistic_regression(full_data):
    v0 = []
    v1 = []
    for i in range(1000, 8000, 500):
        for ycol in range(2,15):
            data = full_data[:i,:]
            test = full_data[-2000:,:]
            y, ypred, yprob, ytest, ytestpred, ytestprob = logistic_regression_test(data, test, ycol)
            print '%5d %3d %7.4f %7.4f' % (i, ycol, compute_accuracy(y, ypred), compute_accuracy(ytest, ytestpred))
        

def historical_mode_benchmark(graphs, paths, scosts, pcosts):
    '''
    A little routine to capture how I've been running logistic
    regression against the dataset and compute the accuracy of it.
    '''
    data = get_shortest_path_matrix(graphs, paths, scosts, pcosts)
    xtest = data[:,2:12]
    ytest = data[:,12]
    modes = np.sum(xtest, 1)
    yprob = modes / 10.0
    ypred = modes > 10
    return ytest, ypred, yprob
    
