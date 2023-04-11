#!/bin/bash
#
#SBATCH --job-name=Power-10
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=10_%a.out
#SBATCH --error=10_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python Power.py 10 $SLURM_ARRAY_TASK_ID
