# kfbook2.py
#
# This is an entry into the Kaggle competition for the Facebook
# mapping the internet competition. This program reads in the provided
# datasets to train a model, and predict the data as noted in the
# competition details.
#

import numpy as np
import loader

train_format = 'clean/train%d.txt'
costs_format = 'cache/scost%d.txt'
file_low = 1
file_high = 15
train_times = 15
test_times = 5
paths_file = 'clean/paths.txt'
submission_file = 'submission.csv'
analysis_file = 'analysis.csv'

def write_submission(pred, filename):
    m, n = pred.shape
    with open(filename, 'w') as f:
        f.write('Probability\n')
        for j in range(n):
            for i in range(m):
                f.write('%f\n' % pred[i,j])

def write_analysis_matrix(data, pred, filename):
    m, nd = data.shape
    m, np = pred.shape
    with open(filename, 'w') as f:
        for i in range(m):
            for j in range(nd):
                f.write('%d,' % data[i,j])
            for j in range(np):
                f.write('%f,' % pred[i,j])
            f.write('\n')

if __name__ == '__main__':

    print 'loading clean training files'
    edge_lists = loader.load_train_files(train_format, file_low, file_high)
    graphs = loader.get_graphs_from_edge_lists(edge_lists)
    
    print 'loading paths file'
    paths = loader.load_paths_file(paths_file)

    print 'loading shortest path files'
    scosts = loader.load_shortest_path_costs(costs_format, file_low, file_high)

    print 'computing path costs'
    pcosts = loader.compute_path_costs(graphs, paths)

    print 'getting shortest path matrix'
    data = loader.get_shortest_path_matrix(graphs, paths, scosts, pcosts)

    print 'running logistic regression'
    ytest, ypred, yprob = loader.logistic_regression(data)
    prob = np.matrix([yprob, yprob, yprob, yprob, yprob]).transpose()

    print 'writing analysis matrix'
    write_analysis_matrix(data, prob, analysis_file)
    
    print 'writing submission'
    write_submission(prob, submission_file)
