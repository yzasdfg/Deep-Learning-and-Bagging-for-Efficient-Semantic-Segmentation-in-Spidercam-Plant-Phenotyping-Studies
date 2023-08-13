#!/usr/bin/env python
# coding: utf-8



# In[1]:
"""
CNN model:deeplabv3+ on MobilenetV2 backbone
training (80%), validation (20%) sample randomly selected from population patches. 
"""

import sys
import os,random, logging
import numpy as np
import glob
import tensorflow as tf

seed= int(sys.argv[4])

"""
	Enable 100% reproducibility on operations related to tensor and randomness.
	Parameters:
	seed (int): seed value for global randomness
	fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
"""
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)    
	

logging.warning("*******************************************************************************")
logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
logging.warning("*******************************************************************************")

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'



session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)



sys.path.append(r'/home/yzstat/yinglunz/UAV_images/')
from helper_func import *
#set_global_determinism()

import joblib
import os
import json
import cv2
from MobilenetV2_keras import Deeplabv3
import pandas as pd
import matplotlib.pyplot as plt
import random, keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler

import datetime
import random
#from deeplab_tf24 import *
path = r'/home/yzstat/yinglunz/Maize_images/'


# In[2]:


learning_rate= float(sys.argv[1])
dropout_rate = float(sys.argv[2])
train_val_size= int(sys.argv[3])

batch_size= int(sys.argv[5])
image_seed=int(sys.argv[6])


print('learning_rate: ', learning_rate)

print('dropout_rate: ', dropout_rate)
print('train_val_size: ', train_val_size)


print('seed: ', seed)
print('batch_size: ', batch_size)
print('image_seed: ', image_seed)
print(path)
weight_dir='dropout_rate_lr' + str(learning_rate)+ '_'+ str(dropout_rate)+'_batch' + str(batch_size)+ '_rs'+str(seed)+'_is' +str(image_seed)

#weight_dir='dropout_rate_lr' + str(learning_rate)+ '_'+ str(dropout_rate)+'_rs'+str(seed)


if not os.path.isdir(weight_dir):
    os.mkdir(weight_dir)
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[3]:


# In[39]:


# # Load data

# In[5]:



#os.chdir(path)
dir_list = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
dir_list


# # CNN 

# ## 1. select at random

# In[6]:

print(path)
image_patch_list = sorted(glob.glob(path+ 'maize_image_patches/train/*.png'))
mask_patch_list = sorted(glob.glob(path+ 'maize_label_patches/train/*.png'))

print('Total train_val size, image and mask: ' , len(image_patch_list), len(mask_patch_list))

# train, val selection, save the selected image patches

train_image_patch_list, val_image_patch_list, train_mask_patch_list, val_mask_patch_list, unselected_image_patch_list= \
train_val_split(image_patch_list, mask_patch_list, train_val_size, image_seed)

unselected_index=[image_patch_list.index(f) for f in unselected_image_patch_list]


selected_index=list(i for i in range(len(image_patch_list)) if i not in unselected_index)

joblib.dump(selected_index, os.path.join(weight_dir, 'selected_index_' +str(train_val_size)+ '.pkl'))
joblib.dump(unselected_index, os.path.join(weight_dir, 'unselected_index_' +str(train_val_size)+ '.pkl'))



joblib.dump(train_image_patch_list, os.path.join(weight_dir, 'train_image_patch_list_' + str(train_val_size) + '.pkl'))
joblib.dump(val_image_patch_list, os.path.join(weight_dir, 'val_image_patch_list_' + str(train_val_size) + '.pkl'))
joblib.dump(unselected_image_patch_list, os.path.join(weight_dir, 'unselected_image_patch_list_' + str(train_val_size) + '.pkl'))


train_image_list=[train_image_patch_list , [1]*len(train_image_patch_list)]
train_image_df = pd.DataFrame(train_image_list).transpose()#, columns =['filename', 'class'])
train_image_df.columns =['filename', 'class']


train_mask_list=[train_mask_patch_list , [1]*len(train_mask_patch_list)]
train_mask_df = pd.DataFrame(train_mask_list).transpose()#, columns =['filename', 'class'])
train_mask_df.columns =['filename', 'class']

val_image_list=[val_image_patch_list , [1]*len(val_image_patch_list)]
val_image_df = pd.DataFrame(val_image_list).transpose()#, columns =['filename', 'class'])
val_image_df.columns =['filename', 'class']


val_mask_list=[val_mask_patch_list , [1]*len(val_mask_patch_list)]
val_mask_df = pd.DataFrame(val_mask_list).transpose()#, columns =['filename', 'class'])
val_mask_df.columns =['filename', 'class']



# In[9]:

# descriptive statistics about the selected training, val samples
# image class count, pixel count

num_classes=2


# In[10]:

num_train_images = len(train_image_patch_list)

num_val_images = len(val_image_patch_list)
print(num_train_images, num_val_images)



mask_count=np.zeros(num_classes).astype('int')
total_count=np.zeros(num_classes).astype('int')
train_image_patch_list_temp=train_image_patch_list.copy()
train_mask_patch_list_temp=train_mask_patch_list.copy()

for i in range(len(train_image_patch_list_temp))[:]:
    mask = cv2.imread(train_mask_patch_list_temp[i], 0)
    count=np.array([np.count_nonzero(mask==i) for i in range(num_classes)])

    #print(count)
    if count[0]==mask.shape[0]*mask.shape[1]:
  
        rand=tf.random.uniform((), seed=42)
        if rand>10:
             train_image_patch_list.remove(train_image_patch_list_temp[i])
             train_mask_patch_list.remove(train_mask_patch_list_temp[i])
        else: 
             mask_count[0]=mask_count[0]+1
             total_count=total_count+count
        

    else:
        total_count=total_count+count
        for j in range(1, len(mask_count)):
            if count[j] >0: mask_count[j]=mask_count[j]+1


print('training image class: ', mask_count)
print('training pixel class: ', total_count)


# In[11]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler


# augmentation, enlarge training size


H, W =mask.shape[0], mask.shape[1]



BUFFER_SIZE = 100
# In[14]:




# In[15]:
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_data = trainGenerator(batch_size,train_image_df,train_mask_df, x_col='filename',y_col='class', aug_dict=data_gen_args,\
                       num_class = num_classes,save_to_dir = None,target_size = (H, W))

data_gen_args=dict()
val_data = trainGenerator(batch_size,val_image_df,val_mask_df, x_col='filename',y_col='class', aug_dict=data_gen_args,\
                       num_class =num_classes,save_to_dir = None,target_size = (H, W))

print('input image shape: ', H, W)

# train the model and save the best weights
model = Deeplabv3(weights='imagenet', input_shape=(H, W,3), classes=num_classes, dropout_rate=dropout_rate,  backbone='mobilenetv2',  alpha=1.)

print('ImageNet weights') 
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# In[16]:




logdir = os.path.join("logs_"+ str(train_val_size), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=logdir, write_graph=True, update_freq='batch')


mc = ModelCheckpoint(mode='min', filepath=os.path.join(weight_dir, 'top_weights_deeplab_MobilenetV2_keras_Maize_generator_'+str(train_val_size)+'.h5'),
                     monitor='val_loss',
                     save_best_only=True,
                     save_weights_only=True, verbose=1)

mc2 = ModelCheckpoint(mode='min', filepath=os.path.join(weight_dir, 'weights_deeplab_MobilenetV2_keras_Maize_generator_'+str(train_val_size)+'.h5'),
                     monitor='val_loss',
                     save_best_only=False,
                     save_weights_only=True, verbose=1)

es=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

callbacks = [tb, mc]


# In[25]:


#steps_per_epoch = len(train_image_patch_list ) // batch_size
#validation_steps =len(val_image_patch_list)// batch_size
steps_per_epoch = min(1000, 4000 // batch_size)  #train_data.samples // batch_size
validation_steps =min(1000, 4000// batch_size)  # val_data.samples// batch_size

import timeit

start = timeit.default_timer()

history=model.fit_generator(train_data ,epochs=50, steps_per_epoch=steps_per_epoch, validation_data=val_data, validation_steps= validation_steps, callbacks=callbacks)

stop = timeit.default_timer()

print('Time: ', stop - start) 

if not os.path.isdir("plots"):
    os.mkdir("plots)
    print('Make directory')

plot_loss_accuracy(history, save=True, filename='./plots/random_sampling_' +str(train_val_size)+ '.png')


#CLOSE TF SESSION
tf.compat.v1.keras.backend.clear_session()
