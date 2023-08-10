'''
@author: masoud
Created on Aug 23, 2020
computes SimRank matrix form S=C·(Q^T·S·Q)+(1−C)·I

If you use this implementation of SimRank, please kindly cite the following paper:
Masoud Reyhani Hamedani and Sang-Wook Kim, On Investigating Both Effectiveness and Efficiency of Embedding Methods in Task of Similarity Computation of
Nodes in Graphs. Applied Sciences, 11(1), 162, 2021, https://doi.org/10.3390/app11010162
'''
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np

def simrank(graph='', iterations=0, topK=0):
    '''
        :graph: a graph file as edgelist (a text file containing one edge per line)
        :iteration: # of total iteration
        :topK: topK results to be written in an output file by descending order    
    '''
    decay_factor = 0.8
    node_set = set ()
    #===========================================================================
        # reading graph
        # NOTE: we can use networkX for reading the graph as well    
    #===========================================================================            
    rows = []; cols = []; sign = [];
    with open(graph, "r") as f:
        lines = f.readlines()
        for line in lines:
            # =================================================
                # the value of cell (node_1,node_2) is set a 1.
            # =================================================
            edge = line.split('\t')
            rows.append(int(edge[0]))
            cols.append(int(edge[1]))
            sign.append(float(1))
            node_set.update((int(edge[0]),int(edge[1])))             
        csr_adj = csr_matrix((sign, (rows, cols)), shape=(len(node_set), len(node_set))) ## --- compressed sparse row representation of adjacency matrix
    f.close()
    print ('The adjacency matrix is compressed in row format ...')
    #===========================================================================
        # column normalizing the sparse adjacency matrix
    #===========================================================================
    norm_csr_adj = normalize(csr_adj, norm='l1', axis=0)
    print ('Column normalization is done ...')
    
    iden_matrix = np.identity(len(node_set),dtype=float)
    iden_matrix = iden_matrix * (1-decay_factor)
    result_ = iden_matrix ## S_0        
    print('===========================================================')
    
    for itr in range (1,iterations+1):
        print ("Iteration {} .... ".format(itr))
        result_ = decay_factor*(norm_csr_adj.transpose() @ result_ @ norm_csr_adj) + iden_matrix

    
    
    
    
