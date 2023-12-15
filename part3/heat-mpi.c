/*
 * Iterative solver for heat distribution
 */
//Added headers
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>

#include <mpi.h>
#include "heat.h"

#define NBx 8

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Common variables
    int resolution, np;
    unsigned iter, maxiter;
    int algorithm;
    double residual = 0.0, resParacial = 0.0;
    
    // algorithmic parameters
    algoparam_t param;
    double runtime, flop;
    
    //Set affinity of each process to a CPU
    int cpuList[20] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuList[myid], &cpuset);
    sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);

    // Master process input data
    if (myid == 0) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

        // check arguments
        if( argc < 2 )
        {
        usage( argv[0] );
        return 1;
        }

        // check input file
        if( !(infile=fopen(argv[1], "r"))  ) 
        {
        fprintf(stderr, 
            "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
        
        usage(argv[0]);
        return 1;
        }

        // check result file
        resfilename= (argc>=3) ? argv[2]:"heat.ppm";

        if( !(resfile=fopen(resfilename, "w")) )
        {
        fprintf(stderr, 
            "\nError: Cannot open \"%s\" for writing.\n\n", 
            resfilename);
        usage(argv[0]);
        return 1;
        }

        // check input
        if( !read_input(infile, &param) )
        {
        fprintf(stderr, "\nError: Error parsing input file.\n\n");
        usage(argv[0]);
        return 1;
        }
        print_params(&param);


        // set the visualization resolution
        param.u     = 0;
        param.uhelp = 0;
        param.uvis  = 0;
        param.visres = param.resolution;
    
        if( !initialize(&param) )
        {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
                return 1;
        }

        // starting time
        runtime = wtime();

        //Save information to send to workers
        maxiter = param.maxiter;
        resolution = param.resolution;
        algorithm = param.algorithm;
        
    }    

    //Send general information to all workers
    MPI_Bcast(&maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&resolution, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&algorithm, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int rowsEach = resolution/numprocs;
    int colsEach = resolution;
    int initRow = (colsEach+2);
    int gostEnd = (colsEach+2) + rowsEach*(colsEach+2);

    int nextProcess = myid + 1;
    if (nextProcess >= numprocs) nextProcess = MPI_PROC_NULL; 
    int prevProcess = myid - 1;
    if (prevProcess < 0) prevProcess = MPI_PROC_NULL;

    //Allocate memory for workers
    double *u;
    double *uhelp;

    iter = 0;
    switch(algorithm) {
        case 0: // JACOBI

            if(myid != 0){
                u = (double*) calloc((rowsEach+2)*(colsEach+2), sizeof(double));
                uhelp = (double*) calloc((rowsEach+2)*(colsEach+2),sizeof(double));
            }else{
                u = param.u;
                uhelp = param.uhelp;
            }

            //Send the pieces of the matrix to each worker
            if (myid == 0) {
                //Send each of the pices of the matrix to each worker
                for(int i = 1; i < numprocs; i++){
                    int firsElement = i*rowsEach*(colsEach+2);
                    int numberElements = (rowsEach+2)*(colsEach+2);
                    MPI_Send(&param.u[firsElement],numberElements, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                }
            }
            else {    
                //Receive the peach from the master
                MPI_Recv(u, (rowsEach+2)*(colsEach+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            }

            int gostInit = 0;
            int endRow = rowsEach*(colsEach+2);
            int lineSize = (colsEach+2);
            MPI_Request r[4];

            while(1) {
                resParacial = relax_jacobi(u, uhelp, rowsEach+2, colsEach+2);

                /* Send last row to next process unless I'm the 1st process */
                MPI_Isend(&uhelp[endRow],lineSize,MPI_DOUBLE,nextProcess,1,MPI_COMM_WORLD, &r[0]);
                
                /* Get last row of the previous process*/
                MPI_Irecv(&u[gostInit],lineSize,MPI_DOUBLE,prevProcess,1,MPI_COMM_WORLD, &r[1]);

                /* Send firsRow*/
                MPI_Isend(&uhelp[initRow],lineSize,MPI_DOUBLE,prevProcess,0,MPI_COMM_WORLD, &r[2]);
                
                /*Recibe firstRow as last gosh row*/
                MPI_Irecv(&u[gostEnd],lineSize,MPI_DOUBLE,nextProcess,0,MPI_COMM_WORLD, &r[3]);

                /*Store uhelp in u*/
                for (int i = 1; i <= rowsEach; i++) {
                    int row = i * (colsEach + 2);
                    for (int j = 1; j <= colsEach; j++) {
                        u[row + j] = uhelp[row + j];
                    }
                }
            
                residual = 0.0;
                MPI_Allreduce(&resParacial,&residual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD ); 

                /*Wait for the communication to finish*/
                MPI_Waitall(4, r, MPI_STATUSES_IGNORE);

                iter++;
                if (residual < 0.00005) break;

                if (maxiter>0 && iter>=maxiter) break;
            }
            break;

        case 1: // RED-BLACK
            residual = relax_redblack(param.u, np, np);
            break;
        case 2: // GAUSS
            //Allocate memory for workers
            if(myid != 0){
                u = (double*) calloc((rowsEach+2)*(colsEach+2), sizeof(double));
            }else{
                u = param.u;
            }

            //---Distribute the matrix to the workers---//
            if (myid == 0) {
                //Send each of the pices of the matrix to each worker
                for(int i = 1; i < numprocs; i++){
                    int firsElement = i*rowsEach*(colsEach+2);
                    int numberElements = (rowsEach+2)*(colsEach+2);
                    MPI_Send(&u[firsElement],numberElements, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
                }
            }
            else {    
                //Receive the pice from the master
                MPI_Recv(u, (rowsEach+2)*(colsEach+2), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            }

            //--Position variables over the matrix--//
            int firstElem, lastElem, lastRow= rowsEach*(colsEach+2);
            int blockSize = colsEach/NBx;

            while(1) {
                resParacial = 0;

                //---Process the blocks reciving allways the upperGost row firs---//
                for(int i = 0; i < NBx; i++){
                    
                    firstElem = i*blockSize + 1;
                    lastElem = firstElem + blockSize - 1;

                    MPI_Recv(&u[firstElem],blockSize,MPI_DOUBLE,prevProcess,0,MPI_COMM_WORLD, &status);
                    
                    resParacial += relax_gauss (u,firstElem,lastElem,rowsEach+2,colsEach+2);    
                    
                    MPI_Send(&u[lastRow + firstElem],blockSize,MPI_DOUBLE,nextProcess,0,MPI_COMM_WORLD);
                }
                
                //---Send of the dowsGost row---//
                MPI_Send(&u[initRow],colsEach+2,MPI_DOUBLE,prevProcess,0,MPI_COMM_WORLD);
                
                MPI_Recv(&u[gostEnd],colsEach+2,MPI_DOUBLE,nextProcess,0,MPI_COMM_WORLD,  &status);

                
                residual = 0.0;
                MPI_Allreduce(&resParacial,&residual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD ); 
                MPI_Barrier(MPI_COMM_WORLD);

                iter++;
                if (residual < 0.00005) break;
                if (maxiter>0 && iter>=maxiter) break;
            }
            break;
        }


    //Gather the pieces of the matrix
    if(myid == 0){
        for (int i = 1; i < numprocs; i++) {
            int firsElement = i*rowsEach*(colsEach+2) + (colsEach+2);
            int numberElements = rowsEach*(colsEach+2);
            MPI_Recv(&u[firsElement], numberElements, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        }
    }
    // Worker process
    else {
        // Send uhelp to master (only the useful part of the matrix)
        MPI_Send(&u[colsEach + 2], rowsEach * (colsEach + 2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);   
    }


    if( myid == 0 ) {
        
        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
            flop/1000000000.0,
            flop/runtime/1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // for plot...
        coarsen( param.u, resolution + 2, resolution + 2,
            param.uvis, param.visres+2, param.visres+2 );
    
        write_image( resfile, param.uvis,  
            param.visres+2, 
            param.visres+2 );

        finalize( &param );
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(0);
}
