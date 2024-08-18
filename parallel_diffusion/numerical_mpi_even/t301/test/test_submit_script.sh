#!/bin/bash
#
#SBATCH --nodes=3
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --partition=compute2011

mpirun python test_diffusion.py
