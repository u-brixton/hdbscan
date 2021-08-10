# -*- coding: utf-8 -*-
"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from warnings import warn

from ._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters)
from ._hdbscan_linkage import label

# Author: Leland McInnes <leland.mcinnes@gmail.com>
#         Steve Astels <sastels@gmail.com>
#         John Healy <jchealy@gmail.com>
#
# License: BSD 3 clause
from numpy import isclose

def _tree_to_labels(X, single_linkage_tree, min_cluster_size=10,
                    cluster_selection_method='eom',
                    allow_single_cluster=False,
                    match_reference_implementation=False,
					cluster_selection_epsilon=0.0,
                    max_cluster_size=0):
    """Converts a pretrained tree and cluster size into a
    set of labels and probabilities.
    """
    condensed_tree = condense_tree(single_linkage_tree,
                                   min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    labels, probabilities, stabilities = get_clusters(condensed_tree,
                                                      stability_dict,
                                                      cluster_selection_method,
                                                      allow_single_cluster,
                                                      match_reference_implementation,
													  cluster_selection_epsilon,
                                                      max_cluster_size)

    return (labels, probabilities, stabilities, condensed_tree,
            single_linkage_tree)

def mst_linkage_core_py(distance_matrix):
    result = np.zeros((distance_matrix.shape[0] - 1, 3))
    node_labels = np.arange(distance_matrix.shape[0], dtype=np.intp)
    current_node = 0
    current_distances = np.infty * np.ones(distance_matrix.shape[0])
    current_labels = node_labels
    
    for i in range(1, node_labels.shape[0]):
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        left = current_distances[label_filter]
        right = distance_matrix[current_node][current_labels]
        current_distances = np.where(left < right, left, right)

        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        result[i - 1, 0] = current_node
        result[i - 1, 1] = new_node
        result[i - 1, 2] = current_distances[new_node_index]
        current_node = new_node
    
    return result


def _hdbscan_prims_kdtree(X, min_samples=5, alpha=1.0,
                          metric='minkowski', p=2, leaf_size=40,
                          gen_min_span_tree=False, **kwargs):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(X, k=min_samples,
                                dualtree=True,
                                breadth_first=True)[0][:, -1].copy(order='C')
    
    distance_matrix = pairwise_distances(X, metric=metric, **kwargs)
    mutual_reachability_ = kdtree_mutual_reachability(X, distance_matrix, metric,
                                                        p, min_points=min_samples, 
                                                        alpha=alpha)

    min_spanning_tree = mst_linkage_core_py(mutual_reachability_)
    # Sort edges of the min_spanning_tree by weight
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]),
                        :]

    # Convert edge list into standard hierarchical clustering format
    single_linkage_tree = label(min_spanning_tree)
    
    if gen_min_span_tree:
        warn('Cannot generate Minimum Spanning Tree; '
             'the implemented Prim\'s does not produce '
             'the full minimum spanning tree ', UserWarning)

    return single_linkage_tree, None

def kdtree_mutual_reachability(X, distance_matrix, metric, p=2, min_points=5,
                               alpha=1.0, **kwargs):
    dim = distance_matrix.shape[0]
    min_points = min(dim - 1, min_points)

    tree = KDTree(X, metric=metric, **kwargs)

    core_distances = tree.query(X, k=min_points)[0][:, -1]

    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    stage1 = np.where(core_distances > distance_matrix,
                      core_distances, distance_matrix)
    result = np.where(core_distances > stage1.T,
                      core_distances.T, stage1.T).T
    return result

def hdbscan_mini(X, min_cluster_size=5, min_samples=None, alpha=1.0, cluster_selection_epsilon=0.0,
            metric='euclidean',
            match_reference_implementation=False,
            algorithm='prims_kdtree',
            max_cluster_size=0, p=None, leaf_size=40,
            approx_min_span_tree=True, gen_min_span_tree=False,
            core_dist_n_jobs=4,
            cluster_selection_method='eom', allow_single_cluster=False, **kwargs):
    """Perform HDBSCAN clustering from a vector array or distance matrix.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    min_cluster_size : int, optional (default=5)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        defaults to the min_cluster_size.

    cluster_selection_epsilon: float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        See [3]_ for more information. Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using approximate_predict), as the approximate_predict function
        is not aware of this argument.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [2]_ for more information.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.

    leaf_size : int, optional (default=40)
        Leaf size for trees responsible for fast nearest
        neighbour queries.

    memory : instance of joblib.Memory or string, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree : bool, optional (default=False)
        Whether to generate the minimum spanning tree for later analysis.

    core_dist_n_jobs : int, optional (default=4)
        Number of parallel jobs to run in core distance computations (if
        supported by the specific algorithm). For ``core_dist_n_jobs``
        below -1, (n_cpus + 1 + core_dist_n_jobs) are used.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to t=True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
        (default False)

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.

    probabilities : ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are assigned
        0.

    cluster_persistence : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.

    single_linkage_tree : ndarray, shape (n_samples - 1, 4)
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

    min_spanning_tree : ndarray, shape (n_samples - 1, 3)
        The minimum spanning as an edgelist. If gen_min_span_tree was False
        this will be None.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [3] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical 
	   Density-based Cluster Selection. arxiv preprint 1911.02282.
    """
    if min_samples is None:
        min_samples = min_cluster_size

    if type(min_samples) is not int or type(min_cluster_size) is not int:
        raise ValueError('Min samples and min cluster size must be integers!')

    if min_samples <= 0 or min_cluster_size <= 0:
        raise ValueError('Min samples and Min cluster size must be positive'
                         ' integers')

    if min_cluster_size == 1:
        raise ValueError('Min cluster size must be greater than one')

    if type(cluster_selection_epsilon) is int:
        cluster_selection_epsilon = float(cluster_selection_epsilon)

    if type(cluster_selection_epsilon) is not float or cluster_selection_epsilon < 0.0:
        raise ValueError('Epsilon must be a float value greater than or equal to 0!')

    if not isinstance(alpha, float) or alpha <= 0.0:
        raise ValueError('Alpha must be a positive float value greater than'
                         ' 0!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    if cluster_selection_method not in ('eom', 'leaf'):
        raise ValueError('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    size = X.shape[0]
    min_samples = min(size - 1, min_samples)

    if min_samples == 0:
        min_samples = 1
    
    (single_linkage_tree, result_min_span_tree) = \
        _hdbscan_prims_kdtree(X, min_samples, alpha,
                                metric, p, leaf_size,
                                gen_min_span_tree, **kwargs)
    return _tree_to_labels(X,
                           single_linkage_tree,
                           min_cluster_size,
                           cluster_selection_method,
                           allow_single_cluster,
                           match_reference_implementation,
						   cluster_selection_epsilon,
                           max_cluster_size) + \
            (result_min_span_tree,)