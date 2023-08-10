# Graph Embedding Methods vs Link-based Similarity Measures in Task of Similarity Computation of Nodes in Graphs

**This repository provides:**
1. Python implementations of the following similarity measures:
 - SimRank (2002, ACM SIGKDD, https://doi.org/10.1145/775047.775126)
 - SimRank* (2013, VLDB Endowment, https://doi.org/10.14778/2732219.2732221) 
- JacSim (2017, Information Sciences, https://doi.org/10.1016/j.ins.2017.06.005)
- JPRank (2019, ACM SAC, https://doi.org/10.1145/3297280.3297331)
- Cosine
2. Datasets
  
## Notes
1. All the codes are implemented in Python 3.7 by Eclipse PyDev.
2. The codes can be easily used in other Python IDs and it is possible to use them via command line by applying small changes. 
3. The implementations of link-based similarity measures are based on their matrix forms, which are **significantly faster** than their component forms.
4. The provided codes for link-based similarity measures can be applied to **both** directed and undirected graphs.
5. The Cosine implementation is based on a matrix/vector multiplication technique, which is **significantly faster** than its conventional implementation.

## Datasets and Graph Structure:
1. BlogCatalog, Cora, and Wikipedia datasets are included. 
2. Each dataset has a “_ground_truth_” folder containing a text file per **each** label where each line indicates a **node id**.
3. A graph is represented as a text file under the _edge list format_ in which, each line corresponds to an edge in the graph, _tab_ is used as the separator, and the node index is started from 0.
   
## Citing:
If you find the provided source codes and datasets useful for your research, please consider citing the following paper:
> Hamedani, M.R.; Kim, S-W. On Investigating Both Effectiveness and Efficiency of Embedding Methods in Task of Similarity Computation of Nodes in Graphs. Applied Sciences. 2021, 11, 162. DOI: https://dx.doi.org/10.3390/app11010162
