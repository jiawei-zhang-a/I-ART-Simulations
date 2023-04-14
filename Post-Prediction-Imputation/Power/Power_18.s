#!/bin/bash
#
#SBATCH --job-name=Power-18
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=18_%a.out
#SBATCH --error=18_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python3 Power.py 18 $SLURM_ARRAY_TASK_ID
