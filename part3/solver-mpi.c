#include "heat.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 8
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int row, rowBefor, rowAfter;
    
    for (int i=1; i<= sizex-2; i++){ 
        row = i*sizey;
        rowBefor = (i-1)*sizey;
        rowAfter = (i+1)*sizey;

        for (int j=1; j<= sizey-2; j++) {
            utmp[row+j]= 0.25 * (u[ row      + (j-1) ]+  // left
                                u[ row      + (j+1) ]+  // right
                                u[ rowBefor + j     ]+  // top
                                u[ rowAfter + j     ]); // bottom
            
            diff = utmp[i*sizey +j] - u[i*sizey + j];
            sum += diff * diff; 
        }
    }

    // printf("Primera en jacobi -> %lf\n", utmp[1*sizey+1]);
    return sum;
}


/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u,unsigned sizex1, unsigned sizex2, unsigned rows, unsigned sizey)
{
    double unew, diff, sum=0.0;
    
    for (int i = 1; i <= rows - 2; i++){
        int row = i*sizey;
        int rowBefor = (i-1)*sizey;
        int rowAfter = (i+1)*sizey;
        for (int j = sizex1; j<= sizex2; j++) {
            unew= 0.25 * (      u[ row	+ (j-1) ]+  // left
                                u[ row	+ (j+1) ]+  // right
                                u[ rowBefor	+ j]+   // top
                                u[ rowAfter	+ j]);  // bottom
            diff = unew - u[row + j];
            sum += diff * diff; 
            u[i*sizey+j]=unew;
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
