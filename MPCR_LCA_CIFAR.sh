#!/bin/bash
#
#SBATCH --job-name=MPCR_1
#SBATCH --output=LCAresults.txt
#
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3200

module load matlab/r2013a
module load slurm

echo "Running MPCR LCA on CIFAR10"


srun matlab -nodisplay -nosplash -nodesktop -r "run('HahnLCA_Dictionary_RGB_CIFAR_Block_KOKO.m')"
