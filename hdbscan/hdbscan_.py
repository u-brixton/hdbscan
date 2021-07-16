# -*- coding: utf-8 -*-
"""
HDBSCAN: Hierarchical Density-Based Spatial Clustering
         of Applications with Noise
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import KDTree
from warnings import warn

from ._hdbscan_linkage import (mst_linkage_core,
                               mst_linkage_core_vector,
                               label)
from ._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores)

from ._hdbscan_reachability import (mutual_reachability)
from .dist_metrics import DistanceMetric

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


def _hdbscan_prims_kdtree(X, min_samples=5, alpha=1.0,
                          metric='minkowski', p=2, leaf_size=40,
                          gen_min_span_tree=False, **kwargs):
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    # The Cython routines used require contiguous arrays
    if not X.flags['C_CONTIGUOUS']:
        X = np.array(X, dtype=np.double, order='C')

    tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    # TO DO: Deal with p for minkowski appropriately
    dist_metric = DistanceMetric.get_metric(metric, **kwargs)

    # Get distance to kth nearest neighbour
    core_distances = tree.query(X, k=min_samples,
                                dualtree=True,
                                breadth_first=True)[0][:, -1].copy(order='C')
    # Mutual reachability distance is implicit in mst_linkage_core_vector
    min_spanning_tree = mst_linkage_core_vector(X, core_distances, dist_metric,
                                                alpha)

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


def hdbscan_mini(X, min_cluster_size=5, min_samples=None, alpha=1.0, cluster_selection_epsilon=0.0,
            metric='euclidean',
            match_reference_implementation=False,
            algorithm='prims_kdtree',
            max_cluster_size=0, p=2, leaf_size=40,
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


# Inherits from sklearn
class HDBSCAN(BaseEstimator, ClusterMixin):
    """Perform HDBSCAN clustering from vector array or distance matrix.

    HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
    with Noise. Performs DBSCAN over varying epsilon values and integrates
    the result to find a clustering that gives the best stability over epsilon.
    This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
    and be more robust to parameter selection.

    Parameters
    ----------
    min_cluster_size : int, optional (default=5)
        The minimum size of clusters; single linkage splits that contain
        fewer points than this will be considered points "falling out" of a
        cluster rather than a cluster splitting into two new clusters.

    min_samples : int, optional (default=None)
        The number of samples in a neighbourhood for a point to be
        considered a core point.

    metric : string, or callable, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=None)
        p value to use if using the minkowski metric.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [3]_ for more information.

    cluster_selection_epsilon: float, optional (default=0.0)
		A distance threshold. Clusters below this value will be merged.
        See [5]_ for more information.

    algorithm : string, optional (default='best')
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``generic``
            * ``prims_kdtree``
            * ``prims_balltree``
            * ``boruvka_kdtree``
            * ``boruvka_balltree``

    leaf_size: int, optional (default=40)
        If using a space tree algorithm (kdtree, or balltree) the number
        of points ina leaf node of the tree. This does not alter the
        resulting clustering, but may have an effect on the runtime
        of the algorithm.

    memory : Instance of joblib.Memory or string (optional)
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    gen_min_span_tree: bool, optional (default=False)
        Whether to generate the minimum spanning tree with regard
        to mutual reachability distance for later analysis.

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
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    prediction_data : boolean, optional
        Whether to generate extra cached data for predicting labels or
        membership vectors few new unseen points later. If you wish to
        persist the clustering object for later re-use you probably want
        to set this to True.
        (default False)

    match_reference_implementation : bool, optional (default=False)
        There exist some interpretational differences between this
        HDBSCAN* implementation and the original authors reference
        implementation in Java. This can result in very minor differences
        in clustering results. Setting this flag to True will, at a some
        performance cost, ensure that the clustering results match the
        reference implementation.

    **kwargs : optional
        Arguments passed to the distance metric

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    probabilities_ : ndarray, shape (n_samples, )
        The strength with which each sample is a member of its assigned
        cluster. Noise points have probability zero; points in clusters
        have values assigned proportional to the degree that they
        persist as part of the cluster.

    cluster_persistence_ : ndarray, shape (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

    condensed_tree_ : CondensedTree object
        The condensed tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    single_linkage_tree_ : SingleLinkageTree object
        The single linkage tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    minimum_spanning_tree_ : MinimumSpanningTree object
        The minimum spanning tree of the mutual reachability graph generated
        by HDBSCAN. Note that this is not generated by default and will only
        be available if `gen_min_span_tree` was set to True on object creation.
        Even then in some optimized cases a tre may not be generated.

    outlier_scores_ : ndarray, shape (n_samples, )
        Outlier scores for clustered points; the larger the score the more
        outlier-like the point. Useful as an outlier detection technique.
        Based on the GLOSH algorithm by Campello, Moulavi, Zimek and Sander.

    prediction_data_ : PredictionData object
        Cached data used for predicting the cluster labels of new or
        unseen points. Necessary only if you are using functions from
        ``hdbscan.prediction`` (see
        :func:`~hdbscan.prediction.approximate_predict`,
        :func:`~hdbscan.prediction.membership_vector`,
        and :func:`~hdbscan.prediction.all_points_membership_vectors`).

    exemplars_ : list
        A list of exemplar points for clusters. Since HDBSCAN supports
        arbitrary shapes for clusters we cannot provide a single cluster
        exemplar per cluster. Instead a list is returned with each element
        of the list being a numpy array of exemplar points for a cluster --
        these points are the "most representative" points of the cluster.

    relative_validity_ : float
        A fast approximation of the Density Based Cluster Validity (DBCV)
        score [4]. The only differece, and the speed, comes from the fact
        that this relative_validity_ is computed using the mutual-
        reachability minimum spanning tree, i.e. minimum_spanning_tree_,
        instead of the all-points minimum spanning tree used in the
        reference. This score might not be an objective measure of the
        goodness of clusterering. It may only be used to compare results
        across different choices of hyper-parameters, therefore is only a
        relative score.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Campello, R. J., Moulavi, D., Zimek, A., & Sander, J. (2015).
       Hierarchical density estimates for data clustering, visualization,
       and outlier detection. ACM Transactions on Knowledge Discovery
       from Data (TKDD), 10(1), 5.

    .. [3] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [4] Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J., 2014. Density-Based Clustering Validation. In SDM
       (pp. 839-847).

    .. [5] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical 
	   Density-based Cluster Selection. arxiv preprint 1911.02282.

    """

    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, max_cluster_size=0,
                 metric='euclidean', alpha=1.0, p=None,
                 algorithm='best', leaf_size=40,
                 approx_min_span_tree=True,
                 gen_min_span_tree=False,
                 core_dist_n_jobs=4,
                 cluster_selection_method='eom',
                 allow_single_cluster=False,
                 prediction_data=False,
                 match_reference_implementation=False,
                 **kwargs):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data

        self._metric_kwargs = kwargs

        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None
        self._outlier_scores = None
        self._prediction_data = None
        self._relative_validity = None

    def fit(self, X, y=None):
        """Perform HDBSCAN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns self
        """

        kwargs = self.get_params()
        # prediction data only applies to the persistent model, so remove
        # it from the keyword args we pass on the the function
        kwargs.pop('prediction_data', None)
        kwargs.update(self._metric_kwargs)

        (self.labels_,
         self.probabilities_,
         self.cluster_persistence_,
         self._condensed_tree,
         self._single_linkage_tree,
         self._min_spanning_tree) = hdbscan_mini(X, **kwargs)

        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        self.fit(X)
        return self.labels_

    @property
    def outlier_scores_(self):
        if self._outlier_scores is not None:
            return self._outlier_scores
        else:
            if self._condensed_tree is not None:
                self._outlier_scores = outlier_scores(self._condensed_tree)
                return self._outlier_scores
            else:
                raise AttributeError('No condensed tree was generated; try running fit first.')

    @property
    def condensed_tree_(self):
        if self._condensed_tree is not None:
            return CondensedTree(self._condensed_tree,
                                 self.cluster_selection_method,
                                 self.allow_single_cluster)
        else:
            raise AttributeError('No condensed tree was generated; try running fit first.')

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            return SingleLinkageTree(self._single_linkage_tree)
        else:
            raise AttributeError('No single linkage tree was generated; try running fit'
                 ' first.')

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            if self._raw_data is not None:
                return MinimumSpanningTree(self._min_spanning_tree,
                                           self._raw_data)
            else:
                warn('No raw data is available; this may be due to using'
                     ' a precomputed metric matrix. No minimum spanning'
                     ' tree will be provided without raw data.')
                return None
        else:
            raise AttributeError('No minimum spanning tree was generated.'
                 'This may be due to optimized algorithm variations that skip'
                 ' explicit generation of the spanning tree.')