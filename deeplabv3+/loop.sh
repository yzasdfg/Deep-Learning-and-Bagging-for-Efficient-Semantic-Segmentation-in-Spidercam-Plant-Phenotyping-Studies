#!/bin/sh
#SBATCH --job-name=loop     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
##SBATCH --qos=short
##SBATCH --partition=gpu 
##SBATCH -p yzstat --gres=gpu 
#SBATCH --output=./slurm_out/slurm-%j.out



# default learning rate 0.001
j=0.5
learning_rate=0.001
type=train
image_seed=42
seed=50  #default=42, used 50

batch_size=16


##i: train_val_size
#for i in 10 50 100 200 400 #600 750 #10 50 600 750 #1000 1623 #  10 #350 400 824  1340 # 
for i in 750
do


####full sequence including 42
#for image_seed in 10 20 30 40 42 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290

for image_seed in $(seq 300 330) 
####for random 0.001

do
for batch_size in 8
do

for learning_rate in 0.001 #gi/AL
do


####train CNN
#sbatch ./slurm_file/test_generator.slurm ${learning_rate} 0.5 $i ${seed} ${batch_size} ${image_seed}


# prediction
#sbatch ./slurm_file/prediction_all.slurm ./dropout_rate_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}/top_weights_deeplab_MobilenetV2_keras_Maize_generator_$i.h5  Multiclass_Prediction/Predict_test_Patches_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}_$i Multiclass_Prediction/Predict_train_Patches_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}_$i Multiclass_Prediction/Predict_val_Patches_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}_$i


#iou_cm, prfasvc_cm
#sbatch ./slurm_file/iou_cm_analysis.slurm $i True ./dropout_rate_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}/${type}_image_patch_list_$i.pkl Multiclass_Prediction/Predict_test_Patches_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}_$i ./iou_pkl_dropout_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed} ${type} 
#sbatch ./slurm_file/prfasvc_cm_analysis.slurm $i True Multiclass_Prediction/Predict_test_Patches_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed}_$i ./iou_pkl_dropout_lr${learning_rate}_0.5_batch${batch_size}_rs${seed}_is${image_seed} 

done 
done
done
done