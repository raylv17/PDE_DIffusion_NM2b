#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --exclusive
#SBATCH --partition=compute2011

mpirun python diffusion.py > out.txt
