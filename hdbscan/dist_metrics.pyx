# !python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

# By Jake Vanderplas (2013) <jakevdp@cs.washington.edu>
# written for the scikit-learn project
# modified for HDBSCAN Dual Tree Boruvka algorithm
# License: BSD

import numpy as np
cimport numpy as np
np.import_array()  # required in order to use C-API

from libc.math cimport fabs, sqrt, exp, cos, pow, log, acos, M_PI

DTYPE = np.double
ITYPE = np.intp


######################################################################
# Numpy 1.3-1.4 compatibility utilities
cdef DTYPE_t[:, ::1] get_memview_DTYPE_2D(
                               np.ndarray[DTYPE_t, ndim=2, mode='c'] X):
    return <DTYPE_t[:X.shape[0], :X.shape[1]:1]> (<DTYPE_t*> X.data)


cdef DTYPE_t* get_vec_ptr(np.ndarray[DTYPE_t, ndim=1, mode='c'] vec):
    return &vec[0]


cdef DTYPE_t* get_mat_ptr(np.ndarray[DTYPE_t, ndim=2, mode='c'] mat):
    return &mat[0, 0]
######################################################################


# First, define a function to get an ndarray from a memory bufffer
cdef extern from "numpy/arrayobject.h":
    object PyArray_SimpleNewFromData(int nd, np.npy_intp* dims,
                                     int typenum, void* data)


cdef inline np.ndarray _buffer_to_ndarray(DTYPE_t* x, np.npy_intp n):
    # Wrap a memory buffer with an ndarray. Warning: this is not robust.
    # In particular, if x is deallocated before the returned array goes
    # out of scope, this could cause memory errors.  Since there is not
    # a possibility of this for our use-case, this should be safe.

    # Note: this Segfaults unless np.import_array() is called above
    return PyArray_SimpleNewFromData(1, &n, DTYPECODE, <void*>x)


# some handy constants
from libc.math cimport fabs, sqrt, exp, pow, cos, sin, asin
cdef DTYPE_t INF = np.inf


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


######################################################################
# metric mappings
#  These map from metric id strings to class names
METRIC_MAPPING = {'euclidean': EuclideanDistance,
                  'l2': EuclideanDistance}


def get_valid_metric_ids(L):
    """Given an iterable of metric class names or class identifiers,
    return a list of metric IDs which map to those classes.

    Examples
    --------
    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])
    >>> sorted(L)
    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    """
    return [key for (key, val) in METRIC_MAPPING.items()
            if (val.__name__ in L) or (val in L)]


######################################################################
# Distance Metric Classes
cdef class DistanceMetric:
    """DistanceMetric class

    This class provides a uniform interface to fast distance metric
    functions.  The various metrics can be accessed via the `get_metric`
    class method and the metric string identifier (see below).

    Examples
    --------

    For example, to use the Euclidean distance:

    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[0, 1, 2],
             [3, 4, 5]])
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics
    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
   
    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=DTYPE, order='c')
        self.mat = np.zeros((1, 1), dtype=DTYPE, order='c')
        self.vec_ptr = get_vec_ptr(self.vec)
        self.mat_ptr = get_mat_ptr(self.mat)
        self.size = 1

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        if self.__class__.__name__ == "PyFuncDistance":
            return (float(self.p), self.vec, self.mat, self.func, self.kwargs)
        return (float(self.p), self.vec, self.mat)

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        if self.__class__.__name__ == "PyFuncDistance":
            self.func = state[3]
            self.kwargs = state[4]
        self.vec_ptr = get_vec_ptr(self.vec)
        self.mat_ptr = get_mat_ptr(self.mat)
        self.size = 1

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : string or class name
            The distance metric to use
        **kwargs
            additional arguments will be passed to the requested metric
        """
        if isinstance(metric, DistanceMetric):
            return metric

        # Map the metric string ID to the metric class
        if isinstance(metric, type) and issubclass(metric, DistanceMetric):
            pass
        else:
            try:
                metric = METRIC_MAPPING[metric]
            except:
                raise ValueError("Unrecognized metric '%s'" % metric)


        return metric(**kwargs)

    def __init__(self):
        if self.__class__ is DistanceMetric:
            raise NotImplementedError("DistanceMetric is an abstract class")

    cdef DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                      ITYPE_t size) nogil except -1:
        """Compute the distance between vectors x1 and x2

        This should be overridden in a base class.
        """
        return -999

    cdef DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                       ITYPE_t size) nogil except -1:
        """Compute the reduced distance between vectors x1 and x2.

        This can optionally be overridden in a base class.

        The reduced distance is any measure that yields the same rank as the
        distance, but is more efficient to compute.  For example, for the
        Euclidean metric, the reduced distance is the squared-euclidean
        distance.
        """
        return self.dist(x1, x2, size)

    cdef int pdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] D) except -1:
        """compute the pairwise distances between points in X"""
        cdef ITYPE_t i1, i2
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return 0

    cdef int cdist(self, DTYPE_t[:, ::1] X, DTYPE_t[:, ::1] Y,
                   DTYPE_t[:, ::1] D) except -1:
        """compute the cross-pairwise distances between arrays X and Y"""
        cdef ITYPE_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return 0

    cdef DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) except -1:
        """Convert the reduced distance to the distance"""
        return rdist

    cdef DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
        """Convert the distance to the reduced distance"""
        return dist

    def rdist_to_dist(self, rdist):
        """Convert the Reduced distance to the true distance.

        The reduced distance, defined for some metrics, is a computationally
        more efficent measure which preserves the rank of the true distance.
        For example, in the Euclidean distance metric, the reduced distance
        is the squared-euclidean distance.
        """
        return rdist

    def dist_to_rdist(self, dist):
        """Convert the true distance to the reduced distance.

        The reduced distance, defined for some metrics, is a computationally
        more efficent measure which preserves the rank of the true distance.
        For example, in the Euclidean distance metric, the reduced distance
        is the squared-euclidean distance.
        """
        return dist

    def pairwise(self, X, Y=None):
        """Compute the pairwise distances between X and Y

        This is a convenience routine for the sake of testing.  For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X : array_like
            Array of shape (Nx, D), representing Nx points in D dimensions.
        Y : array_like (optional)
            Array of shape (Ny, D), representing Ny points in D dimensions.
            If not specified, then Y=X.
        Returns
        -------
        dist : ndarray
            The shape (Nx, Ny) array of pairwise distances between points in
            X and Y.
        """
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Xarr
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Yarr
        cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] Darr

        Xarr = np.asarray(X, dtype=DTYPE, order='C')
        if Y is None:
            Darr = np.zeros((Xarr.shape[0], Xarr.shape[0]),
                            dtype=DTYPE, order='C')
            self.pdist(get_memview_DTYPE_2D(Xarr),
                       get_memview_DTYPE_2D(Darr))
        else:
            Yarr = np.asarray(Y, dtype=DTYPE, order='C')
            Darr = np.zeros((Xarr.shape[0], Yarr.shape[0]),
                            dtype=DTYPE, order='C')
            self.cdist(get_memview_DTYPE_2D(Xarr),
                       get_memview_DTYPE_2D(Yarr),
                       get_memview_DTYPE_2D(Darr))
        return Darr


# ------------------------------------------------------------
# Euclidean Distance
#  d = sqrt(sum(x_i^2 - y_i^2))
cdef class EuclideanDistance(DistanceMetric):
    """Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t size) nogil except -1:
        return euclidean_dist(x1, x2, size)

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t size) nogil except -1:
        return euclidean_rdist(x1, x2, size)

    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) except -1:
        return sqrt(rdist)

    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2


# ------------------------------------------------------------
# SEuclidean Distance
#  d = sqrt(sum((x_i - y_i2)^2 / v_i))
cdef class SEuclideanDistance(DistanceMetric):
    """Standardized Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i \frac{ (x_i - y_i) ^ 2}{V_i} }
    """
    def __init__(self, V):
        self.vec = np.asarray(V, dtype=DTYPE)
        self.vec_ptr = get_vec_ptr(self.vec)
        self.size = self.vec.shape[0]
        self.p = 2

    cdef inline DTYPE_t rdist(self, DTYPE_t* x1, DTYPE_t* x2,
                              ITYPE_t size) nogil except -1:
        if size != self.size:
            with gil:
                raise ValueError('SEuclidean dist: size of V does not match')
        cdef DTYPE_t tmp, d=0
        cdef np.intp_t j
        for j in range(size):
            tmp = x1[j] - x2[j]
            d += tmp * tmp / self.vec_ptr[j]
        return d

    cdef inline DTYPE_t dist(self, DTYPE_t* x1, DTYPE_t* x2,
                             ITYPE_t size) nogil except -1:
        return sqrt(self.rdist(x1, x2, size))

    cdef inline DTYPE_t _rdist_to_dist(self, DTYPE_t rdist) except -1:
        return sqrt(rdist)

    cdef inline DTYPE_t _dist_to_rdist(self, DTYPE_t dist) nogil except -1:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2
