//Added headers
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <sched.h>

#include "heat.h"
#include <omp.h>

cpu_set_t* getHardwareData(int* num_nodes_out) {
    int num_nodes = numa_max_node() + 1;
    *num_nodes_out = num_nodes;

    cpu_set_t* cpus_vec = malloc(num_nodes * sizeof(cpu_set_t));
    if (!cpus_vec) {
        // Handle memory allocation failure if needed
        return NULL;
    }

    for (int i = 0; i < num_nodes; i++) {
        struct bitmask* cpus = numa_allocate_cpumask();
        numa_node_to_cpus(i, cpus);
        CPU_ZERO(&cpus_vec[i]);
        
        for (unsigned long j = 0; j < cpus->size; j++) {
            if (numa_bitmask_isbitset(cpus, j)) {
                CPU_SET(j, &cpus_vec[i]);
            }
        }
        numa_free_cpumask(cpus);
    }

    return cpus_vec;
}

void printCPUsByNode() {
    int num_nodes;
    cpu_set_t* cpus_vec = getHardwareData(&num_nodes);
    if (!cpus_vec) {
        printf("Failed to allocate memory or get CPU data.\n");
        return;
    }

    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d CPUs: ", i);
        int first = 1;
        for (int j = 0; j < CPU_SETSIZE; j++) {
            if (CPU_ISSET(j, &cpus_vec[i])) {
                if (!first) {
                    printf(", ");
                }
                printf("%d", j);
                first = 0;
            }
        }
        printf("\n");
    }

    free(cpus_vec);
}

//Number of blocks 
#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define max(a,b) ( ((a) > (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */

//Basic with standard pragma
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = omp_get_max_threads();
    bx = sizex/nbx;
    nby = 1;
    by = sizey/nby;

    int cpuList[10] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

    // printCPUsByNode();
    
    #pragma omp parallel
    {   
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // int cpu = sched_getcpu();
        // if (cpu == -1) {
        //     perror("sched_getcpu");
        // }

        // printf("Thread %d is running on CPU %d\n", thread, cpu);

        #pragma omp for reduction(+:sum) private(diff) 
        for (int ii=0; ii<nbx; ii++)
            for (int jj=0; jj<nby; jj++) 
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                    for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
                            u[ i*sizey     + (j+1) ]+  // right
                                u[ (i-1)*sizey + j     ]+  // top
                                u[ (i+1)*sizey + j     ]); // bottom
                    diff = utmp[i*sizey+j] - u[i*sizey + j];
                    sum += diff * diff; 
                }
    }   
    // return 0.00004;
    return sum;
}

//V1 (nbx variable)
double relax_jacobi1(double *u, double *utmp, unsigned sizex, unsigned sizey) {
   
    double sum = 0.0;
    int nbx, bx, nby, by;

    // Set number of blocks in x and y direction (assuming square matrices)
    nbx = omp_get_max_threads();
    nby = nbx;

    // Block size in x and y direction
    bx = sizex / nbx;
    by = sizey / nby;

   #pragma omp parallel
    {
        //Private variables
        double sumpar = 0.0;  
        double diff = 0.0;    
        
        //Each thread will process a row of blocks
        int jj = omp_get_thread_num() * by;
        
        for (int ii = 0; ii < nbx; ii++) {
                //Process all the elements in X direction of each block
                for (int i = 1 + ii; i <= min((ii + 1) * bx, sizex - 2); i++) {
                    //Process all the elements in Y direction of each block
                    for (int j = 1 + jj; j <= min(jj + by, sizey - 2); j++) {
                        long row = i * sizey;
                        utmp[row + j] = 0.25 * (u[row + (j - 1)] +    
                                                u[row + (j + 1)] +    
                                                u[(i - 1) * sizey + j] + 
                                                u[(i + 1) * sizey + j]);
                        diff = utmp[row + j] - u[row + j];
                        sumpar += diff * diff;
                    }
                }
        }
        #pragma omp critical
        {
            sum += sumpar;
        }
    }
    return sum;
}

//V2 Only one block in X direction
double relax_jacobi2(double *u, double *utmp, unsigned sizex, unsigned sizey) {
   
    double sum = 0.0;
    int nby, by;

    // Set number of blocks in y direction
    nby = omp_get_max_threads();

    // Calculate the number of rows each thread will process
    by = sizey / nby;

    #pragma omp parallel
    {
        // Private variables
        double sumpar = 0.0;  
        double diff = 0.0;
        int thread = omp_get_thread_num();
        
        // 1. Get the starting row of each thread
        int init = 1 + thread * by;
        
        // 2. Get the ending row of each thread
        int end = (thread != nby - 1) ? (thread + 1) * by : sizey - 2;
        
        // Each thread processes its assigned rows
        for (int i = init; i <= end; i++) {
            long row = i * sizey, rowup = (i - 1) * sizey, rowdown = (i + 1) * sizey;
            for (int j = 1; j <= sizex - 2; j++) {
                utmp[row + j] = 0.25 * (u[row + (j - 1)] +    
                                        u[row + (j + 1)] +    
                                        u[rowup + j] + 
                                        u[rowdown + j]);
                diff = utmp[row + j] - u[row + j];
                sumpar += diff * diff;
            }
        }
        
        // Use critical section for updating the shared sum variable
        #pragma omp critical
        {
            sum += sumpar;
        }
    }
    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss1 (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
                }

    return sum;
}

double relax_gauss2 (double *u, unsigned sizex, unsigned sizey)
{
    double sum = 0.0;
    int nbx, bx, nby, by;

    nbx = omp_get_max_threads();
    bx = sizex/nbx;
    nby = nbx;
    by = sizey/nby;
    
    int cpuList[10] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

    #pragma omp parallel
    {
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        double parsum = 0.0;
        
        #pragma omp single
        {
            for (int ii = 0; ii < nbx; ii++) {
                // Calculate the starting and ending indices for the first and last rows of each block
                int start_first_row = (1 + ii * bx) + 1;
                int end_first_row = start_first_row + (sizey - 2);
                int start_last_row =  min((ii + 1) * bx, sizex - 2) + 1;
                int end_last_row = start_last_row + (sizey - 2);

                // Task with dependency on all elements of the first and last rows of each block
                #pragma omp task firstprivate(ii) \
                depend(inout:u[start_first_row:end_first_row]), \
                depend(inout:u[start_last_row:end_last_row])
                { 
                    double unew, diff;
                    for (int jj = 0; jj < nby; jj++) {    
                        for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++){
                            for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) {
                                unew= 0.25 * (    
                                    u[ i*sizey	+ (j-1) ]+  // left
                                    u[ i*sizey	+ (j+1) ]+  // right
                                    u[ (i-1)*sizey	+ j     ]+  // top
                                    u[ (i+1)*sizey	+ j     ]); // bottom
                                diff = unew - u[i*sizey+ j];
                                parsum += diff * diff; 
                                u[i*sizey+j]=unew;
                                // printf("sum: %f\n", sum);
                            }
                        }
                    }
                    #pragma omp atomic
                    sum += parsum;
                }
            }
        }
    }
    return sum;
}


double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
	
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;

    int blocksFinished[NB];
    for(int i = 0; i < NB; ++i)
	    blocksFinished[i] = 0;

	#pragma omp parallel for schedule(static,1) private(diff,unew) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++){
	    	if(ii > 0){
		    	while(blocksFinished[ii-1] <= jj)
			    {
				    #pragma omp flush
			    }
		    }

            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
                }
	    blocksFinished[ii]++;
		#pragma omp flush
	}

    return sum;
}
