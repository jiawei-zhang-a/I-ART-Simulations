#!/bin/bash
#
#SBATCH --job-name=Power-2
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=30
#SBATCH --output=2_%a.out
#SBATCH --error=2_%a.err

module purge

cd ../../
source venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.9/bin:$PATH
source ~/.bashrc

cd Post-Prediction-Imputation
python3 Power.py 2 $SLURM_ARRAY_TASK_ID
