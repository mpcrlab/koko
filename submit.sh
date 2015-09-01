#!/bin/bash
#
#SBATCH --job-name=MPCR_1
#SBATCH --output=LCAresults.txt
#
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3200

srun matlab -nodisplay -nosplash -nodesktop -r "run('HahnLCA_Dictionary_RGB_CIFAR_Block.m')"
