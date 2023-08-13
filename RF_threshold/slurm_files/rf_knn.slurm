#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=32000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu
#SBATCH --gres=gpu 
#SBATCH --constraint=gpu_32gb
#SBATCH --mail-type=end          # send email when job ends

#SBATCH --output=test-%j.out
module purge
module load anaconda
conda activate mygpu

python Random_Forest_Semantic_Segmention_knn.py $1 $2 $3 $4 $5