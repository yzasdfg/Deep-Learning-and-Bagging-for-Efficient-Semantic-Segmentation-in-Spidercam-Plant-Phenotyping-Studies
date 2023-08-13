#!/bin/bash
#SBATCH --job-name=tf2-random     # create a short name for your job
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=32000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=guest_gpu
#SBATCH --gres=gpu 
#SBATCH --constraint=gpu_32gb
#SBATCH --output=./multiclass/slurm_out/test_generator-%j.out

module purge
module load anaconda
conda activate mygpu

python generator_Maize_random_selection.py $1 $2 $3 $4 $5 $6
