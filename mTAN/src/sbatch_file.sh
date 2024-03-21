#!/usr/bin/env bash
#SBATCH --gpus=1
#SBATCH --partition=NGPU
#SBATCH --exclude=gpu-110



cd /home/yalavarthi/gratif/mTAN/src/


srun -u /home/yalavarthi/miniconda3/envs/linodenet/bin/python $@