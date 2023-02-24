import networkx as nx
import numpy as np

def get_centrality(A):
    G = nx.from_numpy_array(A)
    centrality = nx.betweenness_centrality(G, normalized=True)
    centrality = np.asarray(list(centrality.values()), dtype=float)
    return np.around(centrality, 3)

def get_centrality_all(A, src_domain, num_domain):
    # when we have the physical graph
    G = nx.from_numpy_array(A)
    all_domains = list(range(num_domain))
    tgt_domain = list(set(all_domains) - set(src_domain))
    centrality = nx.betweenness_centrality_subset(G, src_domain, tgt_domain, normalized=True)
    centrality = np.asarray(list(centrality.values()), dtype=float)
    return np.around(centrality, 3)
