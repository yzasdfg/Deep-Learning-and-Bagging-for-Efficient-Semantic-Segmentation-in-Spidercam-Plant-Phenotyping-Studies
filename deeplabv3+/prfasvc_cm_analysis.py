#!/usr/bin/env python
# coding: utf-8

"""
Calculate precision, recall, f1 score, accuracy, specificy and vegetation coverage,
"""

# In[1]:



import os, joblib
import json
import cv2
import numpy as np
import glob
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from helper_func import *

import random
path = r'/home/yzstat/yzhan/Maize_images/'

result_path = r'/work/yzstat/yzhan/Maize_images_MobilenetV2_keras/'
# In[2]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

train_val_size= sys.argv[1]
is_test_iou_cm = sys.argv[2]
#image_list_pkl= sys.argv[3]
pred_test_dir = sys.argv[3]
out_dir = sys.argv[4]
#out_prefix = sys.argv[6]



print('train_val_size: ', train_val_size )
print('is_test_iou_cm : ', is_test_iou_cm)
#print('image_list_pkl: ', image_list_pkl)
#print('val_image_list_pkl: ', val_image_list_pkl)
print('pred_test_dir: ', pred_test_dir)

#print('pred_train_dir: ', pred_train_dir)

print('out_dir: ', out_dir )
#print('out_prefix: ', out_prefix )
  


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    print('Make directory')







#os.chdir(path)
dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
dir_list

num_classes=2
# # CNN 

# In[21]:
# In[ ]:

if is_test_iou_cm=='True':
    test_image_patch_list = sorted(glob.glob(path+ r'maize_image_patches/test/*.png'))
    mgray_list=[]

    pred_test_mask_patch_list=sorted(glob.glob(os.path.join(result_path+pred_test_dir, '*.png')))

    test_mask_patch_list=sorted(glob.glob(path+'maize_label_patches/test/*.png'))

    test_image_patch_files=[os.path.basename(file) for file in test_image_patch_list]

    pred_test_mask_patch_files=[os.path.basename(file) for file in pred_test_mask_patch_list]
    test_mask_patch_files=[os.path.basename(file) for file in test_mask_patch_list]
    assert pred_test_mask_patch_files==test_mask_patch_files, 'pred, ground truth do not match.'
    assert test_image_patch_files==test_mask_patch_files, 'image, mask do not match.'

    print('pred test image size: ', len(pred_test_mask_patch_list))

    all_prfasvc=np.zeros([len(test_mask_patch_list), 7])
    all_mask, all_pred =np.array([]), np.array([])

    for i in range(len(test_mask_patch_list)):
        #print(i)
        #assert os.path.basename(test_mask_patch_list[i])==os.path.basename(pred_mask_patch_list[i])

        img=cv2.imread(test_image_patch_list[i], 0)
        mgray_list.append(np.mean(img))

        mask = tf.io.read_file(test_mask_patch_list[i])
        mask= tf.image.decode_png(mask, channels=1)

        y_pred=tf.io.read_file(pred_test_mask_patch_list[i])
        y_pred= tf.image.decode_png(y_pred, channels=1)


        all_mask=np.append(all_mask, tf.reshape(mask, [-1]))
        all_pred=np.append(all_pred, tf.reshape(y_pred, [-1]))

        mask = tf.reshape(mask, [mask.shape[0]*mask.shape[1]])
        y_pred = tf.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1]])
        
        
        prf=precision_recall_fscore_support(mask, y_pred, pos_label=1, average='binary')[:-1]

        prfa=np.append(np.array(prf), accuracy_score(mask, y_pred)) 
        all_prfasvc[i, :4]=prfa

        cm1 = confusion_matrix(mask, y_pred)
        spec = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        all_prfasvc[i, 4]=spec
        
        vc_mask=np.count_nonzero(mask)/(len(mask))
        vc_pred=np.count_nonzero(y_pred)/(len(y_pred))
        
        all_prfasvc[i, 5]=vc_mask
        all_prfasvc[i, 6]=vc_pred

    joblib.dump(all_prfasvc, os.path.join(out_dir, 'test_prfasvc_' + train_val_size+'.pkl' ))
    joblib.dump(mgray_list, os.path.join(out_dir, 'test_grayness.pkl' ))

    cm_analysis(all_mask, all_pred, os.path.join(out_dir, 'test_cm_' + train_val_size+'.png'), save_file=True )

