#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=12
#SBATCH --output=./Result/Power_%A_%a.out
#SBATCH --error=./Result/Power_%A_%a.err


module purge
mkdir Result

cd ../../
source venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.9/bin:$PATH
source ~/.bashrc

cd Post-Prediction-Imputation
python3 Power.py 