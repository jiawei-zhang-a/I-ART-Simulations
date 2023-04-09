#!/bin/bash
#
#SBATCH --job-name=Level
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=40

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH
source ~/.bashrc

cd ../
python Level.py 2
