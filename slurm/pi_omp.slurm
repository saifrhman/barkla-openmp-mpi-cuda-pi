#!/bin/bash
#SBATCH --job-name=pi_omp
#SBATCH --output=pi_omp_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --partition=short

module load gcc

# Use all assigned CPUs as OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run with 100 million samples (you can change this)
./pi_openmp 100000000
