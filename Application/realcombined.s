#!/bin/bash
#
#SBATCH --job-name=Real
#SBATCH --nodes=1
#SBATCH --time=25:29:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=28


module purge

singularity exec --nv \
    --overlay /scratch/jz4721/pyenv/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; python /scratch/jz4721/Post-prediction-Causal-Inference/Application/MainCombined.py"
