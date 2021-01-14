'''
Created on Aug 6, 2020
computes the SimRank*; 
@author: masoud
'''
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np

def simrank_star(graph='', iterations=0, topK=0):
    '''
        :graph: a graph file as edgelist (a text file containing one edge per line)
        :iteration: # of total iteration
        :topK: topK results to be written in an output file by descending order    
    '''
    decay_factor = 0.8
    node_set = set ()
    #===========================================================================
        # reading graph
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
    del node_set
    iden_matrix = iden_matrix * (1-decay_factor)
    csr_iden_matrix = csr_matrix(iden_matrix)
    del iden_matrix
    result_ = csr_iden_matrix ## S_0            
    print('===========================================================')
    
    #===========================================================================
        # starting iterative computation
    #===========================================================================
    for itr in range (1,iterations+1):
        print ("Iteration {} .... ".format(itr))
        temp_ = result_ @ norm_csr_adj ## --- S_k * Q
        result_ = (decay_factor/2.0)*(temp_.transpose() + temp_) + csr_iden_matrix
        del temp_ 
        write_to_file(result_.todense(), topK, itr)
         


def write_to_file(result_matrix, topK, itr):
    '''
        Writes the results of each iteration in a file.
    '''
    sim_file = open ('SRS_Top'+str(topK)+'_IT_'+str(itr),'w')

    for target_node in range (0,len(result_matrix)):
        target_node_res = result_matrix[target_node].tolist()[0]
        target_node_res_sorted = sorted(target_node_res,reverse=True)
        count = 0
        for index in range (0,len(target_node_res_sorted)):
            val = target_node_res_sorted[index]
            target_node_res[target_node_res.index(val)] = np.nan              
            if val!=0 and target_node_res.index(val)!= target_node:
                sim_file.write(str(target_node)+','+str(target_node_res.index(val))+','+str(round(val,5))+'\n') 
                count = count + 1
                if count == topK:
                    break
    sim_file.close()  
    print ("The result of SimRank*, iteration {} is written in the file!.".format(itr)) 
    print('=============================================================================')


simrank_star(graph="XYZ.txt", 
             iterations=10, 
             topK=30)

