#!/bin/bash

#SBATCH --job-name=submit-mpi.sh
#SBATCH -D .
#SBATCH --output=submit-mpi.sh.o%j
#SBATCH --error=submit-mpi.sh.e%j
## #SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
PROGRAM=heat-mpi

input=test.dat

HOST=$(echo $HOSTNAME | cut -f 1 -d'.')

# if [ ${HOST} = 'boada-6' ] || [ ${HOST} = 'boada-7' ] || [ ${HOST} == 'boada-8' ]
# then
#     echo "Use sbatch to execute this script"
#     exit 0
# fi

USAGE="\n USAGE: ./submit-mpi.sh [numProcess] \n
	numProcess  -> OPTIONAL: Number of process in parallel execution\n
		                    (defaults to using 8 process)\n"

if [ $# != 1 ]
then
	echo -e $USAGE
	procs=8
else
	procs=$1
fi

echo procs=$procs

#make clean
make $PROGRAM

mpirun.mpich -np $procs ./$PROGRAM $input

