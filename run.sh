#!/bin/bash --login

#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --mem=5G  
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%A_%a.out



cd /mnt/ufs18/home-052/surunze/gene_interaction/
module purge
module load GCC/8.3.0
module load Python/3.8.3
module load CUDA/10.2.89
source /mnt/ufs18/home-052/surunze/gene_interaction/co_func_env/bin/activate
cd /mnt/ufs18/home-052/surunze/gene_interaction/VGAE_pyG/

python train.py