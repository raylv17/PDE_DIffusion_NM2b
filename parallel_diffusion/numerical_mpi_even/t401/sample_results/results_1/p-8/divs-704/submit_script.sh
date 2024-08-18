#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --partition=compute2011

mpirun python diffusion.py > out.txt
