#!/bin/sh
#SBATCH --job-name=loop     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=1000       # Maximum memory required per CPU (in megabytes)
#SBATCH --qos=short
#SBATCH --output=./slurm_out/slurm-%j.out




#test_image_index=0


#for UAV data, 2160 images: max point_per_train_image 1000 for k=1, 2 on 32gb GPU, 32000 CPU

# Maize data, 750 training images: max point_per_train_image 2000 for k=1, 2, 3, 4 and max point_per_train_image 3000 for k=1, 2 , 3 

#points 4000 on gpu, 3000 3nn on batch

interval=25
for point_per_train_image in 3000
do
for k in 3
do
#for test_image_index in 10
#do
#######################################################################################
#for test_image_index in $(seq 0 13) #13 ##test patches: 328//interval=328//25=13, [0-13] as starting point
#do
##########################################################################################################
####for comparion of deep learning algorithms. run on gpu ########
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn_gpu.slurm $k $point_per_train_image $test_image_index $interval ./RF_iou_cm_
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn_rgb_gpu.slurm $k $point_per_train_image $test_image_index $interval ./RF_iou_cm_

#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_gpu.slurm $k $point_per_train_image $test_image_index ./RF_iou_cm_

##############################################5 fold cross validation#############################
#for n_estimators in 50 100 150 200
#do
#for max_depth in 10 20 30 40 50
#do
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn_cv.slurm $k $point_per_train_image $n_estimators $max_depth ./RF_iou_cm_\
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn_rgb_cv.slurm $k $point_per_train_image $n_estimators $max_depth ./RF_iou_cm_

#done
#done
#############################################################################################################
### model training and prediction
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn.slurm $k $point_per_train_image $test_image_index $interval ./RF_iou_cm_
#sbatch /work/yzstat/yinglunz/Maize_images_RF/multiclass/slurm_file/rf_knn_rgb.slurm $k $point_per_train_image $test_image_index $interval ./RF_iou_cm_


#done

###iou and prfavc
i=750
#sbatch ./slurm_file/iou_cm_analysis.slurm $i True multiclass_prediction/${k}nn_points_per_image_${point_per_train_image}/RF_${k}nn ./iou_pkl_${k}nn_points_per_image_${point_per_train_image}
#sbatch ./slurm_file/prfasvc_cm_analysis.slurm $i True multiclass_prediction/${k}nn_points_per_image_${point_per_train_image}/RF_${k}nn ./iou_pkl_${k}nn_points_per_image_${point_per_train_image}


sbatch ./slurm_file/iou_cm_analysis.slurm $i True multiclass_prediction/${k}nn_points_per_image_${point_per_train_image}/RF_${k}nn_rgb ./iou_pkl_${k}nn_points_per_image_${point_per_train_image}_rgb
#sbatch ./slurm_file/prfasvc_cm_analysis.slurm $i True multiclass_prediction/${k}nn_points_per_image_${point_per_train_image}/RF_${k}nn_rgb ./iou_pkl_${k}nn_points_per_image_${point_per_train_image}_rgb

##Threshold
lb=10
ub=110
#sbatch ./slurm_file/iou_cm_analysis.slurm $i True multiclass_prediction/threshold_lb${lb}_ub${ub} ./iou_pkl_threshold_lb${lb}_ub${ub}
#sbatch ./slurm_file/prfasvc_cm_analysis.slurm $i True multiclass_prediction/threshold_lb${lb}_ub${ub} ./iou_pkl_threshold_lb${lb}_ub${ub}



done
done
