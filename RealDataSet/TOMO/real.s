#!/bin/bash
#
#SBATCH --job-name=Real
#SBATCH --nodes=1
#SBATCH --time=20:29:00
#SBATCH --mem=250
#SBATCH --cpus-per-task=100

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH

python run.py
