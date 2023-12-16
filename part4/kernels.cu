#include <math.h>
#include <float.h>
#include <cuda.h>

//V1
__global__ void gpu_Heat (float *h, float *g, int N) {

	// Get block and thread inside block for 2D grid
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if( i <= N-2 && j <= N-2 && i >= 1 && j >= 1) {
  		g[i*N+j]= 		0.25 *(h[ i*N     + (j-1) ]+  // left
					           h[ i*N     + (j+1) ]+  // right
				               h[ (i-1)*N + j     ]+  // top
				               h[ (i+1)*N + j     ]); // bottom
 	 }
}


__global__ void gpu_Heat_Reduction(float *h, float *g, float* red, int N) {
    
	//Global identifier
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

	//Local identifier
    int threadId = threadIdx.y * blockDim.x + threadIdx.x; 
    float diff;

    //Reduction array in shared memory
    extern __shared__ float reduction[];
    reduction[threadId] = 0.0;
	if(i == 0 && j == 0){
		*red = 0;
	}

    //Compute the new value
    if(i < N-1 && j < N-1 && i > 0 && j > 0) {
        g[i*N+j] = 0.25 *(h[i*N + (j-1)] + h[i*N + (j+1)] + h[(i-1)*N + j] + h[(i+1)*N + j]);
        diff = g[i*N+j] - h[i*N+j];
        reduction[threadId] = diff * diff;
    }
    
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            reduction[threadId] += reduction[threadId + s];
        }
        __syncthreads();
    }

    // Only the first thread in the block does the atomic add
    if (threadId == 0) {
        atomicAdd(red, reduction[0]);
    }
}





