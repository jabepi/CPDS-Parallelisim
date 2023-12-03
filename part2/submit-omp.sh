#!/bin/bash

#SBATCH --job-name=submit-omp.sh
#SBATCH -D .
#SBATCH --output=submit-omp.sh.o%j
#SBATCH --error=submit-omp.sh.e%j
export OMP_DYNAMIC=false

input=test.dat
PROG=heat-omp
HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

if [ ${HOST} = 'boada-6' ] || [ ${HOST} = 'boada-7' ] || [ ${HOST} == 'boada-8' ]
then
    echo "Use sbatch to execute this script"
    exit 0
fi

USAGE="	\n 
		USAGE: ./submit-omp.sh [numthreads] \n
		numthreads  -> OPTIONAL: Number of threads in parallel execution\n
		                    (defaults to using 8 threads)\n"

if [ $# != 1 ]
then
	echo -e $USAGE
	export OMP_NUM_THREADS=8
else
	export OMP_NUM_THREADS=$1
fi

echo OMP_NUM_THREADS=$OMP_NUM_THREADS
make $PROG

/usr/bin/time -o time-${PROG}-${OMP_NUM_THREADS}-${HOST}.txt ./$PROG $input

