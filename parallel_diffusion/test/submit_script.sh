#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --exclusive
#SBATCH --partition=compute2011

mpirun python test_diffusion.py
