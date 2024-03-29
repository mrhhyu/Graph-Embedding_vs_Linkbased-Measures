'''
@author: masoud
Created on Aug 26, 2020
computes JPRank matrix form

If you use this source code of JPRank, please kindly cite its original paper as follows:
Masoud Reyhani Hamedani and Sang-Wook Kim, Pairwise normalization in SimRank variants: problem, solution, and evaluation. 
In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing, ACM SAC 2019, pp. 534–541, DOI:https://doi.org/10.1145/3297280.3297331
'''
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np


def JPRank(graph='', alpha_in=0.0, alpha_out=0.0, beta=0.0, iterations=0, topK=0):
    '''
        :graph: a graph file as edgelist (a text file containing one edge per line)
        :alpha_in: values of parameter alpha for in-links
        :alpha_out: values of parameter alpha for out-links
        :beta: values of parameter beta
        :iteration: # of total iteration
        :topK: topK results to be written in an output file by descending order    
    '''
    decay_factor = 0.8
    node_set = set ()
    
    #================================================================================================================
        # reading graph; constructing the in-link and out-link sets of each node; compress their adjacency matrices
        # NOTE: we can use networkX for reading the graph as well
    #================================================================================================================
    rows = []; cols = []; sign = []; inlink_dict={}
    rows_out = []; cols_out = []; sign_out = []; outlink_dict={}
    with open(graph, "r") as f:
        lines = f.readlines()
        for line in lines:
            # ===============================================================================
                # in in-links adjacency matrix the value of cell (node_1,node_2) is set a 1.
            # ===============================================================================
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

            # ================================================================================
                # in out-links adjacency matrix the value of cell (node_1,node_2) is set a 1.
            # ================================================================================
            rows_out.append(int(node_2))
            cols_out.append(int(node_1))
            sign_out.append(float(1))            
            if int(node_1) not in outlink_dict:
                out_links = set()
                out_links.add(int(node_2))
                outlink_dict [int(node_1)] = out_links
            else:
                outlink_dict[int(node_1)].add(int(node_2))            
                
    ds_size = len(node_set)                              
    csr_adj_in = csr_matrix((sign, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of in-links adjacency matrix
    csr_adj_out = csr_matrix((sign_out, (rows_out, cols_out)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of out-links adjacency matrix    
    f.close()
    print ('The adjacency matrices are compressed in row format for both in-links and out-links ...')    
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
            if intersection_size!=0: ## -- only non-zero values are stored
                union_size = len(inlink_dict[target_node].union(inlink_dict[node]))
                rows.append(target_node)
                cols.append(node)
                vals.append(intersection_size/float (union_size))
                rows.append(node)
                cols.append(target_node)
                vals.append(intersection_size/float (union_size))

                in_link_pair_dict[(target_node,node)] = (intersection,len(inlink_dict[target_node])*len(inlink_dict[node]))
                
    csr_jaccard_in = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of jaccard matrix    
    rows = []; cols = []; vals = []; 
    out_link_pair_dict = {} ## -- keeps the intersection of out-links sets for each node-pair and their length multiplication for future reference 
    keyList = list (outlink_dict)     
    for target_node_index in range(0, len(keyList)):
        for node_index in range(target_node_index, len(keyList)):
            target_node = keyList[target_node_index]
            node = keyList[node_index]        
            intersection = outlink_dict[target_node].intersection(outlink_dict[node])                
            intersection_size = len(intersection)
            if intersection_size!=0: ## -- only non-zero values are stored
                union_size = len(outlink_dict[target_node].union(outlink_dict[node]))
                rows.append(target_node)
                cols.append(node)
                vals.append(intersection_size/float (union_size))
                rows.append(node)
                cols.append(target_node)
                vals.append(intersection_size/float (union_size))

                out_link_pair_dict[(target_node,node)] = (intersection,len(outlink_dict[target_node])*len(outlink_dict[node]))
                
    csr_jaccard_out = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of jaccard matrix
    print ('Jaccard Coefficient is computed and stored in compressed matrices for both in-links and out-links ...')    

    #===========================================================================
        # column normalizing the sparse adjacency matrices
    #===========================================================================
    norm_csr_adj_in = normalize(csr_adj_in, norm='l1', axis=0)
    norm_csr_adj_out = normalize(csr_adj_out, norm='l1', axis=0)
    iden_matrix = np.identity(ds_size,dtype=float)
    iden_matrix = iden_matrix * (1.0-decay_factor*beta*(alpha_in-alpha_out)-decay_factor*alpha_out)
    result_ = iden_matrix ## S_0
    print ('Column normalization of adjacency matrices and initialization is done ...')
    print('==============================================================================================================')
    
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
                            
        csr_extra_in = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of extra values matrix
        
        rows = []; cols = []; vals = []
        for tople_ in out_link_pair_dict:
            intersection_, multi_ = out_link_pair_dict[tople_][:2]                       
            sum_ = 0
            for outlink_1 in intersection_:
                for outlink_2 in intersection_:
                    sum_ = sum_ + result_[outlink_1,outlink_2]
            rows.append(tople_[0])        
            cols.append(tople_[1])
            vals.append(sum_/(float)(multi_))       
            rows.append(tople_[1])        
            cols.append(tople_[0])
            vals.append(sum_/(float)(multi_))                
                     
        csr_extra_out = csr_matrix((vals, (rows, cols)), shape=(ds_size, ds_size)) ## --- compressed sparse row representation of extra values matrix

        result_ = beta*decay_factor* (alpha_in*csr_jaccard_in + (1.0-alpha_in)*(norm_csr_adj_in.transpose() @ result_ @ norm_csr_adj_in - csr_extra_in)) + \
                  (1.0-beta)*decay_factor* (alpha_out*csr_jaccard_out + (1.0-alpha_out)*(norm_csr_adj_out.transpose() @ result_ @ norm_csr_adj_out - csr_extra_out)) + \
                  iden_matrix
        
