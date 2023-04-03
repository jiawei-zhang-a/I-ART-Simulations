#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=4
#SBATCH --time=6:00:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:rtx8000:1


module purge

cd ../../
source venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.9/bin:$PATH
source ~/.bashrc

cd Post-Prediction-Imputation
python3 Power.py 