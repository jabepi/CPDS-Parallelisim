//Added headers
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <sched.h>

#include "heat.h"
#include <omp.h>

//Number of blocks 
#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define max(a,b) ( ((a) > (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
//V1 WITH COLUMN BLOCKING
double relax_jacobi1 (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = omp_get_max_threads();
    bx = sizex/nbx;
    nby = 1;
    by = sizey/nby;

    int cpuList[20] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    
    #pragma omp parallel
    {   
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

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
    return sum;
}

//V2 WITH OUT COLUMN BLOCKING
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx;
  
    nbx = omp_get_max_threads();
    bx = sizex/nbx;
    
    int cpuList[20] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    
    #pragma omp parallel
    {   
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        #pragma omp for reduction(+:sum) private(diff) 
        for (int ii=0; ii<nbx; ii++)
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                    for (int j=1; j<=sizey-2; j++) {
                    utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
                            u[ i*sizey     + (j+1) ]+  // right
                                u[ (i-1)*sizey + j     ]+  // top
                                u[ (i+1)*sizey + j     ]); // bottom
                    diff = utmp[i*sizey+j] - u[i*sizey + j];
                    sum += diff * diff; 
                }
    }   
    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
//TASK VERSION
double relax_gauss1 (double *u, unsigned sizex, unsigned sizey)
{
    double sum = 0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    
    int cpuList[20] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int dependList[NB][NB];
    
    #pragma omp parallel
    {
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        double parsum = 0.0;
        double unew, diff;
        
        #pragma omp single
        {
            for (int ii = 0; ii < nbx; ii++) {
                for (int jj = 0; jj < nby; jj++) {
                    #pragma omp task firstprivate(ii, jj) \
                    depend(in: dependList[max(ii-1, 0)][jj], dependList[ii][max(jj-1, 0)]) \
                    depend(out: dependList[ii][jj])
                    {
                        for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++){
                            for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                                unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                                        u[ i*sizey	+ (j+1) ]+  // right
                                        u[ (i-1)*sizey	+ j     ]+  // top
                                        u[ (i+1)*sizey	+ j     ]); // bottom
                                diff = unew - u[i*sizey+ j];
                                u[i*sizey+j]=unew;

                                parsum += diff * diff;
                            }
                        }
                        #pragma omp atomic
                        sum += parsum;

                        dependList[ii][jj] = 1;
                    }
                    dependList[0][0] = 1;
                }
            }
        }
    }
    return sum;
}


double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double sum = 0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    
    int cpuList[20] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

    #pragma omp parallel reduction(+:sum)
    {
        //Pin threads to physical cores
        int thread = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuList[thread], &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);


        #pragma omp for ordered(2) 
        for (int ii = 0; ii < nbx; ii++) {
            for (int jj = 0; jj < nby; jj++) {
                #pragma omp ordered depend(sink: ii-1, jj) depend(sink: ii, jj-1)
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++){
                    for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                        double unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
                                u[ i*sizey	+ (j+1) ]+  // right
                                u[ (i-1)*sizey	+ j     ]+  // top
                                u[ (i+1)*sizey	+ j     ]); // bottom
                        double diff = unew - u[i*sizey+ j];
                        u[i*sizey+j]=unew;

                        sum += diff * diff;
                    }
                }
                #pragma omp ordered depend(source)
            }
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
