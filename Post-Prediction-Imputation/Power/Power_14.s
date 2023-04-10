#!/bin/bash
#
#SBATCH --job-name=Power-14
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=40
#SBATCH --output=14_%a.out
#SBATCH --error=14_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python3 Power.py 14 $SLURM_ARRAY_TASK_ID
