'''
@author: masoud
Created on Nov 9, 2020
Vectorized implementation of Cosine

If you use this vectorized implementation of Cosine, please kindly cite its original paper as follows:
Masoud Reyhani Hamedani and Sang-Wook Kim, On Investigating Both Effectiveness and Efficiency of Embedding Methods in Task of Similarity Computation of
Nodes in Graphs. Applied Sciences, 11(1), 162, 2021, https://doi.org/10.3390/app11010162
'''
import numpy as np


def compute_cosine(graph_reps='', topK=''):
    '''
        :graph_reps: a matrix of size (#of nodes * #of dimensions) contains the representation vectors for all nodes
        :topK: topK results to be written in an output file by descending order
        NOTE: 
            Ignore 'RuntimeWarning: invalid value encountered in true_divide' message; it happens when the representation vector of a node contains only '0'.
            This problem is handled during writing the results in the dictionary.
            
    '''
        
    result_dict = {}    
    graph_reps_norm = np.linalg.norm(graph_reps, axis=1)
    
    for row in range (0, len(graph_reps)):
        sim_values={} # dictionary to keep similarity scores between the target node and other nodes (key:node, value:score)
        target_node = graph_reps[row]
        dot_product = graph_reps.dot(target_node)
        target_node_norm = np.linalg.norm(target_node)        
        norms_multiply = np.multiply(graph_reps_norm, target_node_norm)        
        sim_values_all = np.divide(dot_product, norms_multiply)
        
        for node in range (0, len(sim_values_all)):
            if node != row and sim_values_all[node]!=0 and not np.isnan(sim_values_all[node]):
                sim_values[node] = sim_values_all[node]
        sim_values = sorted (sim_values.items(), key=lambda x:x[1],reverse=True)
        if len(sim_values) > topK: ## --- keeping the topK results  
            sim_values = sim_values[:topK]            
        result_dict[row] = sim_values        
        
    sim_file = open ("result.txt",'w')
    for target_node in result_dict:
        for node in result_dict[target_node]:
            sim_file.write(str(target_node)+','+str(node[0])+','+str(round(node[1],5))+'\n')
    print ('The result is written in the file...')
    sim_file.close()
            
            
if __name__=='__main__':

    graph_reps = np.zeros((1055,128))    
    compute_cosine(graph_reps, topK=30)    


