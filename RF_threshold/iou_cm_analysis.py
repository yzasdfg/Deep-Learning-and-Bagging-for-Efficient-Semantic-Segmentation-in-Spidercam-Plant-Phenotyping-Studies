#!/usr/bin/env python
# coding: utf-8

# # UAV data

# In[1]:


import sys
import os, joblib
import json
import cv2
import numpy as np
import glob
import tensorflow as tf

import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split


from helper_func import *

import random
path = r'/home/yzstat/yzhan/Maize_images/'

result_path = r'/home/yzstat/yzhan/Maize_images_RF/'
# In[2]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

train_val_size= sys.argv[1]
is_test_iou_cm = sys.argv[2]

pred_test_dir = sys.argv[3]
out_dir = sys.argv[4]




print('train_val_size: ', train_val_size )
print('is_test_iou_cm : ', is_test_iou_cm)
print('pred_test_dir: ', pred_test_dir)
print('out_dir: ', out_dir )
  


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    print('Make directory')






num_classes=2

# Test IoU
if is_test_iou_cm=='True':
    test_image_patch_list = sorted(glob.glob(path+ r'maize_image_patches/test/*.png'))
    mgray_list=[]

    iou=[]
    intersection, union = np.zeros(num_classes), np.zeros(num_classes)
    all_mask, all_pred =np.array([]), np.array([])
    inter_list, uni_list=[], []


    pred_test_mask_patch_list=sorted(glob.glob(os.path.join(result_path+pred_test_dir, '*.png')))

    test_mask_patch_list=sorted(glob.glob(path+'maize_label_patches/test/*.png'))

    test_image_patch_files=[os.path.basename(file) for file in test_image_patch_list]

    pred_test_mask_patch_files=[os.path.basename(file) for file in pred_test_mask_patch_list]
    test_mask_patch_files=[os.path.basename(file) for file in test_mask_patch_list]
    print('test image size: ', len(test_mask_patch_list))
    print('pred test image size: ', len(pred_test_mask_patch_list))

    assert pred_test_mask_patch_files==test_mask_patch_files, 'pred, ground truth do not match.'
    assert test_image_patch_files==test_mask_patch_files, 'image, mask do not match.'


    for i in range(len(test_mask_patch_list)):
        #print(i)
        #assert os.path.basename(test_mask_patch_list[i])==os.path.basename(pred_mask_patch_list[i])

        img=cv2.imread(test_image_patch_list[i], 0)
        mgray_list.append(np.mean(img))

        mask = tf.io.read_file(test_mask_patch_list[i])
        mask= tf.image.decode_png(mask, channels=1)

        y_pred=tf.io.read_file(pred_test_mask_patch_list[i])
        y_pred= tf.image.decode_png(y_pred, channels=1)

        iou.append(iou_dice(y_pred, mask, num_classes))

        inter, uni=intersection_union(y_pred, mask, num_classes)
        intersection, union=intersection+inter, union+uni

        inter_list.append(inter)
        uni_list.append(uni)

        all_mask=np.append(all_mask, tf.reshape(mask, [-1]))
        all_pred=np.append(all_pred, tf.reshape(y_pred, [-1]))


    joblib.dump(iou, os.path.join(out_dir, 'test_' + train_val_size+'.pkl' ))
    joblib.dump(mgray_list, os.path.join(out_dir, 'test_grayness.pkl' ))
    gen_iou=intersection/union
    joblib.dump(gen_iou, os.path.join(out_dir, 'test_gen_' + train_val_size+'.pkl' ))

    joblib.dump(inter_list, os.path.join(out_dir, 'test_interaction_' + train_val_size+'.pkl' ))
    joblib.dump(uni_list, os.path.join(out_dir, 'test_union_' + train_val_size+'.pkl' ))

    cm_analysis(all_mask, all_pred, os.path.join(out_dir, 'test_cm_' + train_val_size+'.png'), save_file=True )

