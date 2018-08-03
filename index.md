# Introduction

The k-nearest neighbor algorithm (k-NN) is a widely used machine learning algorithm used for both classification and regression. k-NN algorithms are used in many research and industrial domains such as 3-dimensional object rendering, content-based image retrieval, statistics (estimation of entropies and divergences), biology (gene classification), etc. The processing time of the kNN search still remains the bottleneck in many application domains, especially in high dimensional spaces. In this work, we try to address this processing time issue by performing the kNN search using the GPU.


# Authors
	
* Vincent Garcia
* Éric Debreuve
* Michel Barlaud


# Description

Inputs values of the k-NN algorithm:
* Set of reference points.
* Set of query points.
* Parameter k corresponding to the number of neighbors to search for.

For each query point, the k-NN algorithm locates the k closest points (k nearest neighbors) among the reference points set. The algorithm returns (1) the indexes (positions) of the k nearest points in the reference points set and (2) the k associated Euclidean distances.

We provide 3 CUDA implementations for this algorithm:

1. `knn_cuda_global` computes the k-NN using the GPU global memory for storing reference and query points, distances and indexes.

2. `knn_cuda_texture` computes the k-NN using the GPU texture memory for storing the reference points and the GPU global memory for storing other arrays. Using a texture usually speeds-up the computations compared to the first approach. However, due to a size constraint of the texture structures in CUDA, this implementation might not an option for some high dimensional problems.

3. `knn_cublas` computes the k-NN using a different technique involving the use of CUBLAS (CUDA implementation of BLAS). The computation of the distances are split into several sub-problems better suited to a GPU acceleration. As a consequence, on some specific problems, this implementation may lead to a much faster processing time compared to the first two approaches.

For more information about this project, please refer to the original papers listed in the bibliography section bellow.


# Reference compilation / execution

On August 1st, 2018, the code was successfully compiled and executed:

```
PARAMETERS
- Number reference points : 16384
- Number query points     : 4096
- Dimension of points     : 128
- Number of neighbors     : 16

Ground truth computation in progress...

TESTS
- knn_c             : PASSED in 90.46776 seconds (averaged over   2 iterations)
- knn_cuda_global   : PASSED in  0.04166 seconds (averaged over 100 iterations)
- knn_cuda_texture  : PASSED in  0.06296 seconds (averaged over 100 iterations)
- knn_cublas        : PASSED in  0.03112 seconds (averaged over 100 iterations)
```

Specifications of the platform used:
* **OS:** Ubuntu 16.04.1
* **CPU:** Intel Xeon E5-2630 @ 2.3 GHz, 24 cores
* **RAM:** 8 x 4 GB DDR3
* **GPU:** Nvidia Titan X 12GB
  

# Bibliography

* V. Garcia and E. Debreuve and F. Nielsen and M. Barlaud.
  k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching.
  In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010

* V. Garcia and E. Debreuve and M. Barlaud.
  Fast k nearest neighbor search using GPU.
  In Proceedings of the CVPR Workshop on Computer Vision on GPU, Anchorage, Alaska, USA, June 2008.

* Vincent Garcia
  Ph.D. Thesis: Suivi d'objets d'intérêt dans une séquence d'images : des points saillants aux mesures statistiques
  Université de Nice - Sophia Antipolis, Sophia Antipolis, France, December 2008


# Important note

The provided code was written in the context of a Ph.D. thesis. The code was intended to provide a fast and generic solution to the challenging k-NN search problem. However, authors are well aware that for a very specific application (e.g. small number of points, small dimension, only one query point, etc.) there is often a better implementation either in terms of computation time or memory footprint. It is known that a GPGPU implementation should be adapted to the problem and to the specifications of the GPU used. This code will help one to identify if the GPGPU solution is viable for the given problem. Following the terms of the license below, one is allowed and should adapt the code.§


## License

The provided code is licensed under a [Creative Commons Attribution-Share Alike 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/). You are free:
* to Share — to copy, distribute and transmit the work,
* to Remix — to adapt the work,

Under the following conditions:
* Attribution — You must attribute the work in the manner specified by the author or licensor (but not in any way that suggests that they endorse you or your use of the work).
* Noncommercial — You may not use this work for commercial purposes.
* Share Alike — If you alter, transform, or build upon this work, you may distribute the resulting work only under the same or similar license to this one.
