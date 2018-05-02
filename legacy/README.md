AUTHORS
=======
	
	Vincent	Garcia
	Eric	Debreuve
	Michel	Barlaud

	

REFERENCE & BIBTEX
==================

    * V. Garcia and E. Debreuve and F. Nielsen and M. Barlaud.
      k-nearest neighbor search: fast GPU-based implementations and application to high-dimensional feature matching.
      In Proceedings of the IEEE International Conference on Image Processing (ICIP), Hong Kong, China, September 2010

	* V. Garcia and E. Debreuve and M. Barlaud.
	  Fast k nearest neighbor search using GPU.
	  In Proceedings of the CVPR Workshop on Computer Vision on GPU, Anchorage, Alaska, USA, June 2008.
		
	* Vincent Garcia
	  Ph.D. Thesis: Suivi d'objets d'intérêt dans une séquence d'images : des points saillants aux mesures statistiques
	  Université de Nice - Sophia Antipolis, Sophia Antipolis, France, December 2008

		
REQUIREMENTS
============

	- The computer must have a CUDA-enabled graphic card (c.f. NVIDIA website)
	- CUDA has to be installed (CUDA drivers and CUDA toolkit)
	- A C compiler has to be installed
	- For using the CUDA code in Matlab, please refer to the CUDA Matlab plug-in webpage for requirements.


			
COMPILATION & EXECUTION
=======================

	The provided code can be used for C and Matlab applications.
	We provide bellow the C and Matlab procedures to compile and execute our code.
	The user must have a basic knowledge of compiling and executing standard examples before trying to compile and execute our code.
    We will consider here that we want to compile the file "knn_cuda_with_indexes.cu".

	For C
		1.	Set the global variable MATLAB_CODE to 0 in file knn_cuda_with_indexes.cu
		2.	Compile the CUDA file with the command line:
			nvcc -o knn_cuda_with_indexes.exe knn_cuda_with_indexes.cu -lcuda -D_CRT_SECURE_NO_DEPRECATE
		3.	Execute the knn_cuda_with_indexes.exe with the command line:
			./knn_cuda_with_indexes.exe
			
			
	For MATLAB
		1.	Set the global variable MATLAB_CODE to 1 in file knn_cuda_with_indexes.cu
		2.	Compile the CUDA file with the Matlab command line:
			nvmex -f nvmexopts.bat knn_cuda_with_indexes.cu -I'C:\CUDA\include' -L'C:\CUDA\lib' -lcufft -lcudart -lcuda -D_CRT_SECURE_NO_DEPRECATE
		3.	Execute the run_matlab.m script

		
		
ORGANISATION OF DATA
====================
	
	In CUDA, it is usual to use the notion of array.
	For our kNN search program, the following array
		
		A = | 1 3 5 |
		    | 2 4 6 |
	
	corresponds to the a set of 3 points of dimension 2:
	
		p1 = (1, 2)
		p2 = (3, 4)
		p3 = (5, 6)
	
	The array A is actually stored in memory as a linear vector:
	
		A = (1, 3, 5, 2, 4, 6)

	The organisation of data is different in Matlab and in CUDA. For Matlab, the previous linear vector
	corresponds to an array of 3 lines and 2 columns:
	
		    | 1 2 |
		A = | 3 4 |
		    | 5 6 |
