#!/bin/bash
#
#SBATCH --job-name=Power-4
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=4_%a.out
#SBATCH --error=4_%a.err

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python Power.py 4 $SLURM_ARRAY_TASK_ID
