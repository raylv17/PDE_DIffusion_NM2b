#!/bin/bash
#
#SBATCH --nodes=_node
#SBATCH --ntasks=_procs
#SBATCH --exclusive
#SBATCH --partition=compute2011

mpirun python diffusion.py > out.txt
