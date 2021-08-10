import numpy as np
import sys

sys.path.append('hdbscan')
from hdbscan import hdbscan_mini

inp_vecs = [
    "test_vecs/input_vecs1_zh.npy",
    "test_vecs/input_vecs2_dg.npy"
            ]

output_clust = [
    "test_vecs/output_clusters_vecs1_zh.npy",
    "test_vecs/output_clusters_vecs2_dg.npy"
            ]

assert len(inp_vecs) == len(output_clust), \
    "Length of input path and output path lists are not the same"  
hdb_params = {
    'min_cluster_size': 2, 
    'cluster_selection_method': 'eom'
    }

for inp_path, outp_path in zip(inp_vecs, output_clust):
    inp = np.load(inp_path)
    ref_outp = np.load(outp_path)
    assert inp.shape[0] == len(ref_outp), \
    "Length of clustering results should be the same as input num of vectors"
    clust_res = hdbscan_mini(inp, **hdb_params)[0]
    assert np.all(clust_res == ref_outp), \
    f"Clustering result for {inp_path} not same as reference result {outp_path}"

print("All results of clustering are correct (same as reference)")