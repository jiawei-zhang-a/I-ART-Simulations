#!/bin/bash
#
#SBATCH --job-name=Real
#SBATCH --nodes=1
#SBATCH --time=35:29:00
#SBATCH --mem=250G # Ensure you specify 'G' for GB
#SBATCH --cpus-per-task=28

export OMP_NUM_THREADS=1

module purge

# No need to activate the previous venv, we will use Singularity

# Adjust the Singularity exec command below as necessary
# Note: Make sure the paths match where your Singularity image and overlay are located
singularity exec --nv \
    --overlay /scratch/$USER/Post-prediction-Env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/Post-prediction-Causal-Inference/Application/CIMain.py $SLURM_ARRAY_TASK_ID"

