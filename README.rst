This is a simplified and minified version of hdbscan algo for python

code modified from the repo below and belongs to their authors

https://github.com/scikit-learn-contrib/hdbscan


1) To install locally with pip run:

pip install -e .

from root folder
(needed to compile cython modules)

2) To check that clustering results are the same as reference, run:

python check_on_vecs.py



=======
HDBSCAN
=======

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Performs DBSCAN over varying epsilon values and integrates 
the result to find a clustering that gives the best stability over epsilon.
This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
and be more robust to parameter selection.

In practice this means that HDBSCAN returns a good clustering straight
away with little or no parameter tuning -- and the primary parameter,
minimum cluster size, is intuitive and easy to select.

HDBSCAN is ideal for exploratory data analysis; it's a fast and robust
algorithm that you can trust to return meaningful clusters (if there
are any).

Based on the papers:

    McInnes L, Healy J. *Accelerated Hierarchical Density Based Clustering* 
    In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42.
    2017 `[pdf] <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215642>`_

    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013
    
Documentation, including tutorials, are available on ReadTheDocs at http://hdbscan.readthedocs.io/en/latest/ .  
    
Notebooks `comparing HDBSCAN to other clustering algorithms <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>`_, explaining `how HDBSCAN works <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ and `comparing performance with other python clustering implementations <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Benchmarking%20scalability%20of%20clustering%20implementations-v0.7.ipynb>`_ are available.

------
Citing
------

If you have used this codebase in a scientific publication and wish to cite it, please use the `Journal of Open Source Software article <http://joss.theoj.org/papers/10.21105/joss.00205>`_.

    L. McInnes, J. Healy, S. Astels, *hdbscan: Hierarchical density based clustering*
    In: Journal of Open Source Software, The Open Journal, volume 2, number 11.
    2017
    
.. code:: bibtex

    @article{mcinnes2017hdbscan,
      title={hdbscan: Hierarchical density based clustering},
      author={McInnes, Leland and Healy, John and Astels, Steve},
      journal={The Journal of Open Source Software},
      volume={2},
      number={11},
      pages={205},
      year={2017}
    }
    
To reference the high performance algorithm developed in this library please cite our paper in ICDMW 2017 proceedings.

    McInnes L, Healy J. *Accelerated Hierarchical Density Based Clustering* 
    In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42.
    2017


.. code:: bibtex

    @inproceedings{mcinnes2017accelerated,
      title={Accelerated Hierarchical Density Based Clustering},
      author={McInnes, Leland and Healy, John},
      booktitle={Data Mining Workshops (ICDMW), 2017 IEEE International Conference on},
      pages={33--42},
      year={2017},
      organization={IEEE}
    }

---------
Licensing
---------

The hdbscan package is 3-clause BSD licensed. Enjoy.
