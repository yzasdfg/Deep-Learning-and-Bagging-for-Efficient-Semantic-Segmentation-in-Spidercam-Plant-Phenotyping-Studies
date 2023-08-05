#!/usr/bin/env python
# coding: utf-8

# # UAV data

# In[1]:
"""
Predict all image patches and calculate prediction time
"""


import sys
sys.path.append(r'/work/yzstat/yzhan/UAV_images/')

import os, joblib
import json
import cv2
import numpy as np
import glob
import tensorflow as tf
from MobilenetV2_keras import Deeplabv3
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from helper_func import *

import random
path = r'/home/yzstat/yzhan/Maize_images/'
result_path=r'/home/yzstat/yzhan/Maize_images_MobilenetV2_keras/'


# In[2]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

model_weight = sys.argv[1]
test_dir= sys.argv[2]
train_dir= sys.argv[3]
exclude_dir=test_dir.replace('test', 'exclude')



print('model_weight : ', model_weight )
print('test_dir: ', test_dir)
print('train_dir: ', train_dir)

# Create output folder
if not os.path.isdir(result_path+test_dir):
    os.mkdir(result_path+test_dir)
    print('Make directory')


if not os.path.isdir(result_path+train_dir):
    os.mkdir(result_path+train_dir)
    print('Make directory')


exclude_path=result_path+exclude_dir
if not os.path.isdir(exclude_path):
    os.mkdir(exclude_path)
    print('Make directory')






num_classes=2

print(path)
train_image_patch_list = sorted(glob.glob(path+ 'maize_image_patches/train/*.png'))
train_mask_patch_list = sorted(glob.glob(path+ 'maize_label_patches/train/*.png'))
test_image_patch_list = sorted(glob.glob(path+ 'maize_image_patches/test/*.png'))
test_mask_patch_list = sorted(glob.glob(path+ 'maize_label_patches/test/*.png'))
exclude_image_patch_list = sorted(glob.glob(path+ 'maize_exclude_image_patches/*.png'))
exclude_mask_patch_list = sorted(glob.glob(path+ 'maize_exclude_label_patches/*.png'))
print('exclude_image_patch_list', len(exclude_image_patch_list))
# # CNN 

# In[21]:
img = tf.io.read_file(test_image_patch_list[0])       
img= tf.image.decode_png(img, channels=3)
#img.shape[0]

H,  W=img.shape[0],  img.shape[1]

print('input image shape:' , H, W)

# Load model weights
model = Deeplabv3(weights=None, input_shape=(H, W,3), classes=num_classes, dropout_rate=0.5,  backbone='mobilenetv2',  alpha=1.)

'''
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.load_weights(model_weight)
#model.load_weights(r'/lustre/work/yzstat/yzhan/UAV_images_MobilenetV2_keras/multiclass/dropout_rate_lr0.001_0.5_batch4_rs50_is42/top_weights_deeplab_MobilenetV2_keras_UAV_generator_2160.h5', by_name=True)
#print('load weights')

model.summary()



# Prediction and record predict time

import timeit

start = timeit.default_timer()

# train (train+val)
for file in train_image_patch_list[:]:
    img = tf.io.read_file(file)
    img= tf.image.decode_png(img, channels=3)
    img=tf.image.resize(images=img, size=[H, W])/255
    #img=cv2.imread(file)
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    img =img[tf.newaxis,...]#.astype('float32')
    y_pred= model.predict(img)
    y_pred=np.argmax(y_pred, axis=-1)[0]
    filename = os.path.basename(file)
    tf.keras.preprocessing.image.save_img(os.path.join(result_path+train_dir, filename), y_pred[..., tf.newaxis], data_format=None, file_format='png', scale=False)
    #cv2.imwrite(os.path.join(result_path+train_dir, filename), y_pred[..., tf.newaxis])

# test
for file in test_image_patch_list[:]:
    img = tf.io.read_file(file)
    img= tf.image.decode_png(img, channels=3)
    img=tf.image.resize(images=img, size=[H, W])/255
    #img=cv2.imread(file)
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    img =img[tf.newaxis,...]#.astype('float32')
    y_pred= model.predict(img)
    y_pred=np.argmax(y_pred, axis=-1)[0]
    filename = os.path.basename(file)
    tf.keras.preprocessing.image.save_img(os.path.join(result_path+test_dir, filename), y_pred[..., tf.newaxis], data_format=None, file_format='png', scale=False)
    #cv2.imwrite(os.path.join(result_path+test_dir, filename), y_pred[..., tf.newaxis])


# exclude image patches
for file in exclude_image_patch_list[:]:
    img = tf.io.read_file(file)
    img= tf.image.decode_png(img, channels=3)
    img=tf.image.resize(images=img, size=[H, W])/255
    #img=cv2.imread(file)
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    img =img[tf.newaxis,...]#.astype('float32')
    y_pred= model.predict(img)
    y_pred=np.argmax(y_pred, axis=-1)[0]
    filename = os.path.basename(file)
    dirname=os.path.basename(os.path.dirname(file))
    

    #if not os.path.isdir(os.path.join(exclude_path, dirname)):
     #   os.mkdir(os.path.join(exclude_path, dirname))
      #  print('Make directory')
    
    #tf.keras.preprocessing.image.save_img(os.path.join(os.path.join(exclude_path, dirname), filename), \
    tf.keras.preprocessing.image.save_img(os.path.join(exclude_path, filename), \
                                          y_pred[..., tf.newaxis], data_format=None, file_format='png', scale=False)
    #cv2.imwrite(os.path.join(result_path+exclude_dir, filename), y_pred[..., tf.newaxis])



stop = timeit.default_timer()

print('Time: ', stop - start) 
