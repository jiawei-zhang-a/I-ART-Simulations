#!/bin/bash
#
#SBATCH --job-name=CI
#SBATCH --nodes=1
#SBATCH --time=01:29:00
#SBATCH --mem=250MB
#SBATCH --cpus-per-task=1
#SBATCH --output=Runtime/%a.out
#SBATCH --error=Runtime/%a.err

export OMP_NUM_THREADS=1

module purge

singularity --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/Post-prediction-Causal-Inference/Simulation/CalculatePowerForSingleOutcomeComplete.py $SLURM_ARRAY_TASK_ID"

