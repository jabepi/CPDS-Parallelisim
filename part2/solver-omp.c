#include "heat.h"
#include <omp.h>
#include <stdio.h>


//Number of blocks 
#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */

//V1
double relax_jacobi2(double *u, double *utmp, unsigned sizex, unsigned sizey) {
    double diff, sum = 0.0;
    int nbx, bx, nby, by;

    // Set number of blocks in x and y direction (assuming square matrices)
    nbx = omp_get_max_threads();
    nby = nbx;

    // Block size in x and y direction
    bx = sizex / nbx;
    by = sizey / nby;

    #pragma omp parallel private(diff) reduction(+:sum)
    {
        double sumpar = 0.0;  // Local sum for each thread
                            // Distribute the rows of blocks among threads

        //Get the row of blocks that each thread is going to process
        int ii; 
        int jj = omp_get_thread_num();  

        //Each threads process all the horizontal blocks of its row
        for (ii = 0; ii < nbx - 1; ii++) {
            //Each threads process all the elements of its block 
            for (int i = 1 + ii * by; i <= (ii + 1) * by; i++) {
                long row = i * sizey;
                for (int j = 1 + jj * bx; j <= min((jj + 1) * bx, sizex - 2); j++) {
                    utmp[row + j] = 0.25 * (u[row + (j - 1)] +    
                                            u[row + (j + 1)] +    
                                            u[(i - 1) * sizey + j] + 
                                            u[(i + 1) * sizey + j]);
                    diff = utmp[row + j] - u[row + j];
                    sumpar += diff * diff;
                }
            }            
        }
        //Each threads process all the elements of its block 
        for (int i = 1 + ii * by; i <= sizey - 2; i++) {
            long row = i * sizey;
            for (int j = 1 + jj * bx; j <= min((jj + 1) * bx, sizex - 2); j++) {
                utmp[row + j] = 0.25 * (u[row + (j - 1)] +    
                                        u[row + (j + 1)] +    
                                        u[(i - 1) * sizey + j] + 
                                        u[(i + 1) * sizey + j]);
                diff = utmp[row + j] - u[row + j];
                sumpar += diff * diff;
            }
        }

        sum += sumpar; 
    }

    return sum;
}

//V2
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum = 0.0;
    int bx, nby, by;

    nby = omp_get_max_threads();;
    by = sizey/nby;

    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int jj = 0; jj < nby; jj++) 
        for (int i = 1; i <= sizex-2; i++) 
            for (int j = 1 + jj*by; j <= min((jj+1)*by, sizey-2); j++) {
                utmp[i*sizey+j] = 0.25 * (u[ i*sizey     + (j-1) ] +  // left
                                        u[ i*sizey     + (j+1) ] +  // right
                                        u[ (i-1)*sizey + j     ] +  // top
                                        u[ (i+1)*sizey + j     ]); // bottom
                diff = utmp[i*sizey+j] - u[i*sizey + j];
                sum += diff * diff; 
            }

    return sum;
}

//V3


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
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
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

