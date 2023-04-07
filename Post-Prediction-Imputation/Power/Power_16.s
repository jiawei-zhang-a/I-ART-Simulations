#!/bin/bash
#
#SBATCH --job-name=Power-16
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=30
#SBATCH --output=16_%a.out
#SBATCH --error=16_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python3 Power.py 16 $SLURM_ARRAY_TASK_ID
