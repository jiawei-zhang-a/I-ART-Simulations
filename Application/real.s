#!/bin/bash
#
#SBATCH --job-name=Real
#SBATCH --nodes=1
#SBATCH --time=09:29:00
#SBATCH --mem=250
#SBATCH --cpus-per-task=28
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err

export OMP_NUM_THREADS=1

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH

python run.py 
