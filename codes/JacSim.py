'''
@author: masoud
Created on Aug 23, 2020
computes JacSim matrix form

If you use this source code of JacSim, please kindly cite its original paper as follows:
Masoud Reyhani Hamedani and Sang-Wook Kim, JacSim: An accurate and efficient link-based similarity measure in graphs. 
Information Sciences, Vol. 414, 2017, pp. 203–224, https://doi.org/10.1016/j.ins.2017.06.005
'''
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np

def JacSim_MF(graph='', alpha=0.0, iterations=0, topK=0):
    '''
        :graph: a graph file as edgelist (a text file containing one edge per line)
        :alpha: values of parameter alpha 
        :iteration: # of total iteration
        :topK: topK results to be written in an output file by descending order    
    '''
    decay_factor = 0.8
    node_set = set ()
    
    #============================================================================================
        # reading graph; constructing the in-link set of each node; compress the adjacency matrix
        # NOTE: we can use networkX for reading the graph as well
    #============================================================================================
    rows = []; cols = []; sign = []; inlink_dict={}
    with open(graph, "r") as f:
        lines = f.readlines()
        for line in lines:
            # =================================================
                # the value of cell (node_1,node_2) is set a 1.
            # =================================================
            node_1, node_2 = line.split('\t')[:2]
            rows.append(int(node_1))
            cols.append(int(node_2))
            sign.append(float(1))
            node_set.update((int(node_1),int(node_2)))            
            if int(node_2) not in inlink_dict:
                in_links = set()
                in_links.add(int(node_1))
                inlink_dict [int(node_2)] = in_links
            else:
                inlink_dict[int(node_2)].add(int(node_1))
    ds_size = len(node_set)                          
    csr_adj = csr_matrix((sign, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of adjacency matrix
    f.close()
    print ('The adjacency matrix is compressed in row format ...')
    
    #========================================================================================================
        # Computing the Jaccard Coefficient for all node pairs; saving them in a compressed symmetric matrix.
    #========================================================================================================
    rows = []; cols = []; vals = []; 
    in_link_pair_dict = {} ## -- keeps the intersection of in-links sets for each node-pair and their length multiplication for future reference
    keyList = list (inlink_dict) 
    for target_node_index in range(0, len(keyList)):
        for node_index in range(target_node_index, len(keyList)):
            target_node = keyList[target_node_index]
            node = keyList[node_index]        
            intersection = inlink_dict[target_node].intersection(inlink_dict[node])                
            intersection_size = len(intersection)
            if intersection_size!=0: ## only non-zero values are stored
                union_size = len(inlink_dict[target_node].union(inlink_dict[node]))
                rows.append(target_node)
                cols.append(node)
                vals.append(intersection_size/float (union_size))

                rows.append(node)
                cols.append(target_node)
                vals.append(intersection_size/float (union_size))

                in_link_pair_dict[(target_node,node)] = (intersection,len(inlink_dict[target_node])*len(inlink_dict[node]))
                
    csr_jaccard = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of jaccard matrix
    print ('Jaccard Coefficient for all nodes is computed and stored in a compressed matrix  ...')    

    #===========================================================================
        # column normalizing the sparse adjacency matrix
    #===========================================================================
    norm_csr_adj = normalize(csr_adj, norm='l1', axis=0)
    iden_matrix = np.identity(ds_size,dtype=float)
    iden_matrix = iden_matrix * (1.0-decay_factor*alpha)
    result_ = iden_matrix ## S_0
    print ('Column normalization of adjacency matrix and initialization is done ...')
    print('==============================================================================================')
    
    ### --- starting the iterative computation 
    for itr in range (1,iterations+1):
        print ("Iteration {} .... ".format(itr))
        #===========================================================================
            # Calculating the extra values for intersection part of in-links 
        #===========================================================================
        rows = []; cols = []; vals = []
        for tople_ in in_link_pair_dict:
            intersection_, multi_ = in_link_pair_dict[tople_][:2]                       
            sum_ = 0
            for inlink_1 in intersection_:
                for inlink_2 in intersection_:
                    sum_ = sum_ + result_[inlink_1,inlink_2]
            rows.append(tople_[0])        
            cols.append(tople_[1])
            vals.append(sum_/(float)(multi_))

            rows.append(tople_[1])        
            cols.append(tople_[0])
            vals.append(sum_/(float)(multi_))                
                            
        csr_extra = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of extra values matrix
        result_ = decay_factor*( alpha*csr_jaccard + (1.0-alpha)*(norm_csr_adj.transpose() @ result_ @ norm_csr_adj - csr_extra) ) + iden_matrix
        write_to_file(result_, alpha, topK, itr)    


def write_to_file(result_matrix, alpha, topK, itr):
    '''
        Writes the results of each iteration in a file.
    '''
    sim_file = open ('JacSim_A_'+str(alpha*10).split('.')[0]+'_Top'+str(topK)+'_IT_'+str(itr),'w')

    for target_node in range (0,len(result_matrix)):
        target_node_res = result_matrix[target_node].tolist()[0]
        target_node_res_sorted = sorted(target_node_res,reverse=True)[:topK+1]
        for val in target_node_res_sorted:
            node = target_node_res.index(val)
            if val!=0 and node!= target_node:
                sim_file.write(str(target_node)+','+str(node)+','+str(round(val,5))+'\n')
            target_node_res[node] = np.nan 
    sim_file.close()  
    print ("The result of JacSim matrix form, iteration {} is written in the file!.".format(itr)) 
    print('==============================================================================================')


for alpha_val in np.arange(0.1,0.2,0.1): ## --- defines the rage of alpha as min, max, step      
    JacSim_MF(graph="XYZ.txt", 
              alpha=alpha_val,
              iterations=4, 
              topK=30)
    

    

    
    
    
    
