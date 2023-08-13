#!/bin/bash
#SBATCH --job-name=tf2-prfasvc      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=64000       # Maximum memory required per CPU (in megabytes)
#SBATCH --partition=gpu
#SBATCH --gres=gpu 

#SBATCH --output=。/multiclass/slurm_out/prfasvc_cm-%j.out

module purge
module load anaconda
conda activate mygpu

python prfasvc_cm_analysis.py $1 $2 $3 $4 $5 $6