# CUDA Conway's Game of Life
### Luan (Remi) Ta: Independent final project - CPSC 5600 (Parallel Computing)
#### Seattle University


## Dependencies:
* NVCC (CUDA): Requires for compilation on supported architectures with compute capability of at least 2.0
* OpenGL (FreeGlut): glut32.dll, glut32.lib and glut.h must be present in the working directory or dynamically linked to the project. Recommended: NVIDIA's CG package.


## Compilation:
* To compile, run the following command on the command prompt: 

		> nvcc -o final gol.cu --use_fast_math
		
* Can also add the -O3 to observe a significant improvement in serial speed (with negligible improvement for the CUDA version):

		> nvcc -o final gol.cu --use_fast_math -O3
		
* Execute the generated executable and wait for the program to generate data for serial & CUDA versions.


## Limitations:
* The program is restricted by array indexing of at most 2^32 elements. As such, the grid size and the number of generations combined must be smaller than 2^32
* When the app starts up, DO NOT move the window or interract with it - this will crash the program!!! The OpenGL graphics code is a quick hack - very fragile!


## Acknowledgements:
* This project is done as part of the final project for Prof. Kevin Lundeen's CPSC 5600: Parallel Computing
* This project references one line of code (to determine whether the cell is alive or dead) from this article: http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA/2-Basic-implementation


## Copyrights:
* No restriction. Do as you're pleased.