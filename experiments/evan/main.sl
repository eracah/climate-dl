#!/bin/bash -l
#SBBATCH -N 1
#SBATCH -p regular
#SBATCH -t 1:00:00
#SBATCH -o slurm_outputs/slurm-%A.out
#SBATCH --qos=premium

module load deeplearning
python $@
