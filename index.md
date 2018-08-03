### Presentation

The [k-nearest neighbor (kNN)](http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm) search is a problem found in many research and industrial domains such as 3-dimensional object rendering, content-based image retrieval, statistics (estimation of entropies and divergences), biology (gene classification), etc. In spite of some improvements in the last decades, the computation time required by the kNN search still remains the bottleneck of methods based on kNN, especially in high dimensionnal spaces.

We propose in this website two GPGPU implementations of the brute-force (exhaustive) kNN search algorithm. These implementations, through the APIs NVIDIA CUDA and NVIDIA CUBLAS, paralellize the kNN search process into the computer graphic card. Thanks to the parallel structure of the GPU and the optimization of the APIs, the proposed implementation appears to be much faster, especially in high dimension and for large dataset, than the celebrated and highly optimized [ANN C++ library](http://www.cs.umd.edu/~mount/ANN/).

### Authors

* [Vincent Garcia](http://www.vincentgarcia.org/)
* [Eric Debreuve](http://www.i3s.unice.fr/~debreuve/)
* [Michel Barlaud](http://www.i3s.unice.fr/~barlaud/)

### Source code information

The provided CUDA code is usable for C, C++ and Matlab. A preprocessor constant in the file's header allows to activate or deactivate the Matlab wrapper. Please read the README file to learn how to adapt the code. We provide 2 implementations of the kNN search to fit to your needs:

* **KNN CUDA** is a CUDA implementation of the k-nearest neighbor search. The maximum number of points is 65535. There is no limit for the dimension except for the memory space it takes.
* **KNN CUBLAS** is a CUBLAS implementation of the k-nearest neighbor search. This version is usually faster than KNN CUDA and is based on matrix multiplication to compute the distances between points.  The maximum number of points is 65535 and the maximum dimension is 8192.

Most applications need to identify the kNN points by their index. Some other applications however are only interested in the distances to the k-th nearest neighbor (e.g. entropy estimation). We thus provide 2 specific implementations:

* **Implementation with indexes** provides for each query point the distances to the k nearest neighbors and the indexes of these neighbors.
* **Implementation without indexes** provides for each query point only the distance to the k-th nearest neighbor. This version is faster than the version with indexes because first it uses less memory and second it does not spent time on processing indexes.

In total, we provide 4 specific implementations for solving the kNN search problem.

#### Important note

The provided code was written in the context of a Ph.D. thesis. The code was intended to provide a fast and generic solution to the challenging kNN search problem. However, authors are well aware that for a very specific application (e.g. small number of points, small dimension, only one query point, etc.) there is often a better implementation either in terms of computation time or memory footprint. It is known that a GPGPU implementation should be adapted to the problem and to the specifications of the GPU used. This code will help one to identify if the GPGPU solution is viable for the given problem. Following the terms of the license below, one is allowed and should adapt the code.

### References

Following the terms of the license below, you need to acknowledge the use of this code. Do so by referencing the following articles.

* V. Garcia and E. Debreuve and F. Nielsen and M. Barlaud. k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching. In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010. [View preprint](http://www.vincentgarcia.org/data/Garcia_2010_ICIP.pdf).
* V. Garcia and E. Debreuve and M. Barlaud. Fast k nearest neighbor search using GPU. In Proceedings of the CVPR Workshop on Computer Vision on GPU, Anchorage, Alaska, USA, June 2008. [View preprint](http://www.vincentgarcia.org/data/Garcia_2008_CVGPU.pdf).
* Vincent Garcia. Ph.D. Thesis: Suivi d'objets d'intérêt dans une séquence d'images : des points saillants aux mesures statistiques Université de Nice - Sophia Antipolis, Sophia Antipolis, France, December 2008.  [View thesis](http://www.vincentgarcia.org/data/Garcia_2008_PHD.pdf).

## License

The provided code is licensed under a [Creative Commons Attribution-Share Alike 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/). You are free:
* to Share — to copy, distribute and transmit the work,
* to Remix — to adapt the work,

Under the following conditions:
* Attribution — You must attribute the work in the manner specified by the author or licensor (but not in any way that suggests that they endorse you or your use of the work).
* Noncommercial — You may not use this work for commercial purposes.
* Share Alike — If you alter, transform, or build upon this work, you may distribute the resulting work only under the same or similar license to this one.
