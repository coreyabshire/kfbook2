from collections import defaultdict, deque

import loader
import networkx as nx

from itertools import permutations

def total_graph_nodes(graphs):
    return sum(len(g) for g in graphs) * 2

def total_path_nodes(paths):
    return sum(len(p) for p in paths)

def total_possible_nodes(graphs, paths):
    return total_graph_nodes(graphs) + total_path_nodes(paths)

def sorted_node_list(nodes):
    return sorted(nodes.items(), key=nodes.get, reverse=True)

def sorted_words(s):
    return ' '.join([''.join(sorted(w)) for w in s.split(' ')])

def append_sorted_words_column(snl):
    return [(a,b,(a.isdigit() and a or sorted_words(a))) for a,b in snl]

def sorted_word_lookup(nodes):
    lookup = {}
    for node, count in nodes.items():
        key = node.isdigit() and node or sorted_words(node)
        if key in lookup:
            lookup[key].append((node,count))
        else:
            lookup[key] = [(node,count)]
    return lookup

def write_nodes(nodes, filename):
    a = sorted(nodes.items(), key=lambda x: x[0])
    with open(filename, 'w') as f:
        for x in a:
            f.write('%s\t%d\n' % x)

def write_asnl(asnl, filename):
    with open(filename, 'w') as f:
        for x in asnl:
            f.write('%s\t%d\t%s\n' % x)

def deletes(name):
    w = name.split(' ') # words in the name
    e = [] # edits of w
    s = [(w[:i], w[i:]) for i in range(len(w) + 1)] # all splits of words
    e += [a + b[1:] for a, b in s if b] # delete
    return set(' '.join(x) for x in e)

def edits(name):
    w = name.split(' ') # words in the name
    e = [] # edits of w
    s = [(w[:i], w[i:]) for i in range(len(w) + 1)] # all splits of words
    e += [a + b[1:] for a, b in s if b] # delete
    e += [a + [b[1]] + [b[0]] + b[2:] for a, b in s if len(b) > 1] # transpose
    e += [a + [x] + b for x in w for a, b in s] # insert
    return set(' '.join(x) for x in e)

def suggest(nodes, lookup, name):
    candidates = set()
    sname = sorted_words(name)
    candidates.add(name)
    candidates.add(sname)
    for e in edits(sname):
        if e in lookup:
            for c in lookup.get(e):
                candidates.add(c[0])
    return max(candidates, key=nodes.get)

def make_correction_map(nodes, lookup):
    final = {}
    for n in nodes:
        r = suggest(nodes, lookup, n)
        if r in final:
            final[r] += nodes[n]
        else:
            final[r] = nodes[n]
    for n in final.keys():
        for d in deletes(n):
            if d in final and n in final:
                if final[n] > final[d]:
                    final[n] += final[d]
                    del final[d]
    return final

def write_with_lookup(nodes, lookup, filename):
    with open(filename, 'w') as f:
        for n in nodes:
            r = suggest(nodes, lookup, n)
            f.write('%s\t%d\t%s\t%d\n' % (n, nodes[n], r, nodes[r]))

def get_nodes(graph):
    nodes = defaultdict(lambda: defaultdict(int))
    for a, b, c in graph:
        nodes[a][b] = c
    return nodes

def breadth_first_search(graph, src, dst, maxcost=2):
    q = [deque() for i in range(maxcost)]
    q[0].append([src])
    for cost in range(maxcost):
        print '    cost level %d' % cost
        while len(q[cost]) > 0:
            path = q[cost].popleft()
            curr_step = path[0]
            if curr_step == dst:
                return [step for step in reversed(path)], cost
            else:
                for next_step, step_cost in graph[curr_step].items():
                    if next_step not in path:
                        if cost + step_cost < maxcost:
                            q[cost + step_cost].append([next_step] + path)
    return None, None

# given a node name, find the node associated with that name

# step 1: train node name decoder
# step 2: load network using node name decoder
# step 3: train network analyzer
# step 4: predict network path optimality
