#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=10
#SBATCH --time=5:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=12
#SBATCH --output=Power_%A_%a.out
#SBATCH --error=Power_%A_%a.err

module purge
module load python/intel/3.8.6

cd ../
python Power.py Data/$SLURM_ARRAY_TASK_ID.txt