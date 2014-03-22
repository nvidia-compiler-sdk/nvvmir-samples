This is a collection of simple NVVM IR programs, each of which illustrates
the use of one GPU specific feature. 

* cuda-shared-memory

  contains a few NVVM IR programs that show how to use CUDA 'shared' memory.

* syscalls

  contains a few NVVM IR programs that show how to make calls to device side 
  malloc/free/vprintf functions. 


* device-side-launch

  contains a complete program that shows how to launch a kernel within a
  kernel (CUDA dynamic parallelism). 

  An installation of CUDA toolkit version 5 or above is required to build
  the program. 

* uvmlite

  contains a complete program that shows how to use the unified virtual 
  memory. It requires >= CUDA 6.0.
