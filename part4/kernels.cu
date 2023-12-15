#include <math.h>
#include <float.h>
#include <cuda.h>


__global__ void gpu_Heat (float *h, float *g, int N) {

	// Get block and thread inside block for 2D grid
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
 	if( i <= N-1 && j <= N-1 && i >= 1 && j >= 1) {
  		g[i*N+j]= 		0.25 *(h[ i*N     + (j-1) ]+  // left
					           h[ i*N     + (j+1) ]+  // right
				               h[ (i-1)*N + j     ]+  // top
				               h[ (i+1)*N + j     ]); // bottom
 	 }
}



// __global__ void gpu_Heat(float *h, float *g, double* residual, int N) {
//     // Shared memory to accumulate block-level residual
//     __shared__ float blockResidual;

//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     // Initialize block residual to 0 for the first thread in the block
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         blockResidual = 0.0;
//     }
//     __syncthreads(); // Synchronize threads within a block

//     // Compute value and residual for valid grid points
//     if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
//         float newValue = 0.25 * (h[(i - 1) * N + j] + h[(i + 1) * N + j] + h[i * N + j - 1] + h[i * N + j + 1]);
//         float localResidual = newValue - h[i * N + j];

//         // Atomic add to accumulate the residual within the block
//         atomicAdd(&blockResidual, abs(localResidual));
        
//         // Update grid point value
//         g[i * N + j] = newValue;
//     }

//     __syncthreads(); // Synchronize threads within a block again

//     // Use one thread to update the global residual
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         atomicAdd(residual, blockResidual);
//     }
// }
