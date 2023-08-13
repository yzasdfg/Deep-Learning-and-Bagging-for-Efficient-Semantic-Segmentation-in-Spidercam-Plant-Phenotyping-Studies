# Maize_images
Prediction_all.py
Predict all image patches
iou_cm_analysis.py
Calculate IoU, cufusion matrix and other statistics
prfasvc_cm_analysis.py
Calculate precision, recall, f1 score, accuracy, specificy and vegetation coverage

## RF_threshold:
### Random_Forest_Semantic_Segmention_knn_cv.py
RF hyper-parameter cross validation
### Random_Forest_Semantic_Segmention_knn.py
RF prediction
### Maize_RF_threshold.ipynb
RF and threshold results

### loop_RF_thre.sh
Call slurm in slurm_files to run .py on hpc

### slurm_files
slurm files that envoke py files


## deeplabv3+

### generator_Maize_random_selection.py
Run deeplabv3+

### Maize_images_CNN.ipynb
Standalone deeplabv3+ and sample size trajectory

### CNN_bagging.ipynb
Deeplabv3+ with bagging

### loop.sh
Call slurm in slurm_files to run .py on hpc

### slurm_files
slurm files that envoke py files

