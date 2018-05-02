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


# Compilation and testing

We provide a test code with the purpose of verifying the correct compilation and execution of the code as well as evaluating the processing time. This code performs the following tasks:
* Create a set of reference points randomly initialized.
* Create a set of query points randomly initialized.
* Compute the ground-truth k-NN using a non-optimized C implementation.
* For each CUDA implementation listed above:
  * Compare the output (indexes and distances) with the ground-truth.
  * Measure the processing time.

One can change the number of points, the point dimension and the parameter k by editing the `test.cpp` file. However, with some parameters value, the provided code might generate errors. For instance, if the number of points is too big, the GPU memory might not be big enough to store the different arrays allocated by the program. Consequently, an allocation error should be displayed.

Before attempting to compile this code, one should verify the following points:
* Computer used has a CUDA-enabled graphic card (c.f. NVIDIA website).
* CUDA drivers and CUDA toolkit are installed.
* C/C++ compiler is available.
* Command `nvcc` installed.
* Command `make` installed.

Compile the project using the Makefile:
* Open a command line tool or a terminal.
* Go to the `code` directory.
* Compile the code using the following command:
  `$ Make`
* Test the code using the following command:
  `$ ./test`


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


# Data storage and representation

The storage and representation of a set of points used in this project is similar to the one found with Deep Neural Network projects. Let us consider the following set of 3 points of dimension 2:
	
```
p1 = (1, 2)
p2 = (3, 4)
p3 = (5, 6)
```

We represent these points as a 2D array A of size 2 x 3:
	
```
A = | 1 3 5 |
    | 2 4 6 |
```
	
Internally, this array is actually stored as a linear array (row-major order):

```
A = (1, 3, 5, 2, 4, 6)
```
  

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


# Legacy code

The original code written in 2008 is still available in the `legacy` folder. 

Changes made in the 2018 update:

- **Code cleaning:** The code is now simpler, cleaner and better documented. 

- **Code simplification:** Back in 2008, 4 implementations were available, but the differents implementations were performing different tasks. The 2018 implementations all compute the same thing: distances and indexes of the k-NN. One can now select more easily the most appropriate implementation to her/his needs.

- **No query points split:** When computing the k-NN using an exhaustive algorithm, a distance matrix (containing the all distances between query and reference points) must be computed. This matrix can be enormous and may not fit into the GPU memory. In the 2008 code, the query points were eventually splitted into different subsets. These subsets were processed iteratively and the results were finally merged. This allowed to partially solve the memory problem as long as the reference points could fit into the GPU memory. However, this part of the code was quite long and complex. We recall that a CUDA implementation should be adapted to the problem dimension and to the hardware employed. We believe the split used in the 2008 version belongs to the code adaptation and should not be provided into a generic framework. Removing this split is consistent with the idea of making the code cleaner, simpler and more readable.

- **Testing:** Thanks to the test code provided in the 2018 update, one can now easily verify that the code compiles smoothly and that the results are correct. By modifying the parameters used in the `test.cpp` file, one can quickly identify the most adapted implementation based on the available options and processing times.

- **Easy compilation:** Using a Makefile makes the project compilation much simpler than before.

- **No Matlab bindings** Back in 2008, Matlab was to go-to programming language in Academy. Nowadays, other options are available. Since we do not want to provide and maintain the bindings for Matlab, Python, etc., we decided to simply remove the initial Matlab bindings. If necessary, these bindings are still available in the legacy code.
