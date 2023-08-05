#!/usr/bin/env python
# coding: utf-8

# # UAV data

# In[1]:
"""
record prediction time for threshold method
"""

import sys
sys.path.append(r'/work/yzstat/yinglunz/UAV_images/')

import os, joblib
import json
import cv2
import numpy as np
import glob
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


from helper_func import *

import random, time
path = r'/work/yzstat/yinglunz/Maize_images/'

result_path = r'/work/yzstat/yinglunz/Maize_images_RF/'
# In[2]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# img = tf.io.read_file(test_image_patch_list[0])       
# img= tf.image.decode_png(img, channels=3)
# img.shape[0]

# # Load data

# In[5]:



#os.chdir(path)


num_classes=2
# # CNN 

# In[21]:
# In[ ]:


test_image_patch_list = sorted(glob.glob(path+ r'maize_image_patches/test/*.png'))

print('test image size: ', len(test_image_patch_list))

test_image_patch_files=[os.path.basename(file) for file in test_image_patch_list]



print('pred test image size: ', len(test_image_patch_list))



all_image_patch_list= test_image_patch_list

all_image_patch_files=[os.path.basename(f) for f in all_image_patch_list]



random.seed(42) 
lb, ub=10, 110
out_dir=result_path+'/multiclass_prediction/threshold_lb'+str(lb)+\
                                                   '_ub'+str(ub)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    print('Make directory')
	
for bounds in list(product([lb],  [ub])): 

    print(bounds)
    #lb, ub=bounds[0], bounds[1]
    #              inds[(rs)*n_images_per_pdf:(rs+1)*n_images_per_pdf]]
    #sample_images=random.sample(all_image_patch_files, 1)
    for i, file in enumerate(all_image_patch_files):
        if i%200==0:
            print(i)
        #print(all_image_patch_files.index(file))
        #mask=cv2.imread(all_mask_patch_list[all_image_patch_files.index(file)], 0)


        #pred_mask=cv2.imread(all_pred_mask_patch_list[all_image_patch_files.index(file)], 0)
        #pred_mask=cv2.imread(all_pred_mask_patch_list[all_image_patch_files.index(file)], 0)
        filename=os.path.basename(file)
        #print(os.path.basename(all_pred_mask_patch_list[all_image_patch_files.index(file)]))
        #print(os.path.basename(all_mask_patch_list[all_image_patch_files.index(file)]))
        #print(np.unique(pred_mask))
        img=cv2.imread(all_image_patch_list[all_image_patch_files.index(file)])
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = (lb,0,0)
        upper = (ub,255,255)
        # Threshold the HSV image
	
        start_time = time.time()

        pred_mask =cv2.inRange(hsv_img, lower, upper)//255
        tf.keras.preprocessing.image.save_img(os.path.join(out_dir, filename), \
                                              pred_mask[..., tf.newaxis], \
                                      data_format=None, file_format='png', scale=False)
        print("--- model prediction time: %s seconds ---" % (time.time() - start_time))

