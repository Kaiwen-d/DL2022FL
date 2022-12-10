#!/bin/bash

## Team ID
#SBATCH --account=csci_ga_2572_2022fa_12

#SBATCH --job-name=torch
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=32GB

#SBATCH --time=1-00:00:00
#SBATCH --output=mae-day1.out
#SBATCH --error=mae-day1.err
#SBATCH --exclusive
#SBATCH --requeue


# pretrain with default settings
singularity exec --nv \
--overlay /scratch/kd1860/Deep_Learning/overlay_11-15.ext3:ro \
-B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ \
-B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
-B /scratch -B /scratch_tmp \
/scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python main_pretrain.py"