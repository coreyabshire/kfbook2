# clean_edits2.py
#
# Reads in a shorter names file and just tries to clean that.
#

from collections import defaultdict
import numpy as np
import loader

def get_name_found_verbose(names, word_lookup):
    name_found = {}
    ordered_names = sorted(names.items(), key=lambda x: x[1], reverse=True)
    for name, count in ordered_names:
        if name in name_found:
            #print '%s already done' % name
            continue
        else:
            print 'checking %s (%d)' % (name, names[name])
        words = name.split()
        for i in range(len(words)):
            for j in range(len(words)):
                new_words = words[:j] + [words[i]] + words[j:]
                new_name = ' '.join(new_words)
                if new_name in names and new_name not in name_found:
                    print '  I: %s (%d)' % (new_name, names[new_name])
                    name_found[new_name] = name
        for i in range(len(words)):
            new_words = words[:i] + words[i+1:]
            new_name = ' '.join(new_words)
            if new_name in names and new_name not in name_found:
                print '  D: %s (%d)' % (new_name, names[new_name])
                name_found[new_name] = name
        for i in range(len(words)):
            k = loader.sort_word(words[i])
            for w, c in word_lookup[k]:
                if w != words[i]:
                    new_words = words[:i] + [w] + words[i+1:]
                    new_name = ' '.join(new_words)
                    if new_name in names and new_name not in name_found:
                        print '  S: %s (%d)' % (new_name, names[new_name])
                        name_found[new_name] = name
    return name_found

def get_name_found(names, word_lookup):
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
            k = loader.sort_word(words[i])
            for w, c in word_lookup[k]:
                if w != words[i]:
                    new_words = words[:i] + [w] + words[i+1:]
                    new_name = ' '.join(new_words)
                    if new_name in names and new_name not in name_found:
                        name_found[new_name] = name
    return name_found

if __name__ == '__main__':
    names = {}
    with open('names_short2.txt') as infile:
        lines = [name.strip() for name in infile.readlines()]
        for line in lines:
            parts = line.split('\t')
            names[parts[0]] = int(parts[1])
    ninstances = len(names)
    print 'read %d instances' % ninstances
    
    for name, count in names.items():
        print '%4d %s' % (count, name)


    word_lookup = loader.make_word_lookup(names)

    ordered_names = sorted(names.items(), key=lambda x: x[1], reverse=True)
    
    print
    for k,v in ordered_names:
        print '%4d %s' % (v, k)

    print
    name_found, edits = loader.get_name_found(names, word_lookup)
    for name in edits.keys():
        print '%s edits:' % name
        for kind, new_name, c1, c2 in edits[name]:
            print '  %s %3d %3d %s' % (kind, c1, c2, new_name)
    

    # find edges by correcting words
    for name in names:
        words = name.split()
        
