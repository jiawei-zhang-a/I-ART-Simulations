#!/bin/bash
#
#SBATCH --job-name=Multiple
#SBATCH --nodes=1
#SBATCH --time=29:29:00
#SBATCH --mem=250
#SBATCH --cpus-per-task=1
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err
export OMP_NUM_THREADS=1

module purge

source /scratch/jz4721/Post-prediction-Causal-Inference/venv/bin/activate
export PATH=/scratch/jz4721/Post-prediction-Causal-Inference/venv/lib64/python3.8/bin:$PATH

cd ../
python CalculatePowerForMultipleOutcome.py $SLURM_ARRAY_TASK_ID
