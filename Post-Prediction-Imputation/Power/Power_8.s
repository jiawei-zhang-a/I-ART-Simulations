#!/bin/bash
#
#SBATCH --job-name=Power-1
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=40
#SBATCH --array=1-250
#SBATCH --output=8_%a.out
#SBATCH --error=8_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python3 Power.py 8 $SLURM_ARRAY_TASK_ID
