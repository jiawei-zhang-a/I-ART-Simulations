#!/bin/bash
#
#SBATCH --job-name=Power
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --time=5:00
#SBATCH --mem=12GB
#SBATCH --cpus-per-task=12
#SBATCH --output=Power_%A_%a.out
#SBATCH --error=Power_%A_%a.err


module purge
module load python/intel/3.8.6
cd ../../
source venv/bin/activate

cd Post-Prediction-Imputation
python Power.py 