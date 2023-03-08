import numpy as np
from sklearn.metrics import pairwise_distances
from DiffusionEMD import DiffusionTree
import graphtools

# Use diffusion EMD to get the matrix
def emd(x_seq_feat, feat_size, n_distributions, n_points_per_distribution):
    x_seq_feat = x_seq_feat.reshape(-1, feat_size)
    x_seq_feat, indices = np.unique(x_seq_feat, axis=0, return_index=True)
    minx = np.min(x_seq_feat, axis=0)
    maxx = np.max(x_seq_feat, axis=0) + 1e-10
    # Normalize
    std_X = (x_seq_feat - minx) / (maxx - minx)
    dc = DiffusionTree(max_scale=10, delta=1e-10, min_basis=20)
    data_graph = graphtools.Graph(std_X, use_pygsp=True, n_pca=100)
    group_ids = np.repeat(np.eye(n_distributions), n_points_per_distribution, axis=0)
    group_ids = group_ids[indices]
    embeds = dc.fit_transform(data_graph.W, group_ids)
    dis_matrix = pairwise_distances(embeds)
    return dis_matrix