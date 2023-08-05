#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RF CV for tunning hyperparameters
Due to the computing time limitation, just run one scenario at a time instead of a squence of scenarios.
Submit multiple jobs with different hyperparameter combinations as input for grid search
"""
import os
import json
import cv2
from tqdm import tqdm
import numpy as np
import glob
from numpy.random import seed
from numpy.random import randint
import joblib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys
import time
import random, tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV

from itertools import product
sys.path.append(r'/home/yzstat/yinglunz/UAV_images/')
from helper_func import *


import random
input_path  = r'/work/yzstat/yinglunz/Maize_images/'

rs=42
k= int(sys.argv[1])
n_points= int(sys.argv[2])
n_estimators=int(sys.argv[3])
max_depth= sys.argv[4]
out_dir_prefix = sys.argv[5]



print('k nearest neighbor: ', k)
print('points per training image: ', n_points)
print('max_depth: ', max_depth)
print('n_estimators: ', n_estimators)
print('out_dir_prefix: ', out_dir_prefix )


# Maximum number of levels in tree
if max_depth!= 'None':
    max_depth = int(max_depth) 
else: 
    max_depth=None


# Number of trees in random forest

n_estimators = int(n_estimators)


# Create the random grid
parameters = {'n_estimators': [n_estimators], 'max_depth': [max_depth]} #, 'random_state': [0]}


out_dir=out_dir_prefix + str(k)+'nn_points_per_image_'+str(n_points)

pred_dir='/home/yzstat/yinglunz/Maize_images_RF/multiclass_prediction/'+str(k)+'nn_points_per_image_'+str(n_points)+'/RF_'+str(k)+'nn'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    print('Make directory')

if not os.path.isdir(os.path.dirname(pred_dir)):
    os.mkdir(os.path.dirname(pred_dir))
    print('Make directory')

if not os.path.isdir(pred_dir):
    os.mkdir(pred_dir)
    print('Make directory')



# In[44]:



image_patch_list = sorted(glob.glob(input_path+ 'maize_image_patches/train/*.png'))
mask_patch_list = sorted(glob.glob(input_path+ 'maize_label_patches/train/*.png'))


image_patch_files=[os.path.basename(file) for file in image_patch_list]


test_image_patch_list = sorted(glob.glob(input_path+ 'maize_image_patches/test/*.png'))
test_mask_patch_list = sorted(glob.glob(input_path+ 'maize_label_patches/test/*.png'))



# In[235]:


num_classes=2
H, W=256, 256
#n_points=10**5



# # Nearest Neighbors Random Forest

# In[244]:

all_points=list(product(range(k, H+2*k-k),  range(k, W+2*k-k))) #shape H+2k, 2+2k after padding 

neighbor= [i for i in range(-k, k+1)]
print(neighbor)
coordinate=list(product(neighbor, neighbor))
print(coordinate)


# In[248]:

seed(rs)
random.seed(rs)

# construct training samples, R, G, B, GLCM values at k neighbors and n_points per image
X_train, y_train=[], []


for i in range(len(image_patch_list))[:]:     
    mask=cv2.copyMakeBorder(cv2.imread(mask_patch_list[i], 0),k,k,k,k,cv2.BORDER_REPLICATE)  
    img=cv2.copyMakeBorder(cv2.imread(image_patch_list[i]),k,k,k,k,cv2.BORDER_REPLICATE)  
    gray_img=cv2.copyMakeBorder(cv2.imread(image_patch_list[i], 0),k,k,k,k,cv2.BORDER_REPLICATE)
    #mask=cv2.imread(mask_patch_list[i], 0)
    #img=cv2.imread(image_patch_list[i])
    #gray_img= cv2.imread(image_patch_list[i], 0)

    points = random.sample(all_points, n_points)

    if i%500==0: print(i)
    #point = randint(k, 512-k, (n_points, 2)) # sample int from [low, high)
    for p in points:
        #print(p)

        #RGB features
        neighbor_img=[img[p[0]+c[0], p[1]+c[1], :] for c in coordinate]
        flat_list = [item for sublist in neighbor_img for item in sublist]

        #GLCM features
        sub_gray_img=gray_img[p[0]+coordinate[0][0]: p[0]+coordinate[-1][0],  p[1]+coordinate[0][1]: p[1]+coordinate[-1][1]]
        #bins = np.linspace(0, 255, 64)        # gray level:64
        #compress_gray = np.digitize(img, bins)
        #gray = np.uint8(compress_gray) 
        mean, _=list_glcm(sub_gray_img, d=[k])  # data type of the image£ºuint8

        #combine the two features
        flat_list.extend(mean)
        #print(len(neighbor_img))
        #print(neighbor_img)
        '''
        if flat_list in X_train:
            if y_train[X_train.index(flat_list)]==mask[point[p][0], point[p][1]]: continue
        '''
        X_train.append(flat_list)
        y_train.append(mask[p[0], p[1]])





print('input feature shape: ', np.array(X_train).shape)
#joblib.dump(X_train, os.path.join(out_dir, 'RF_'+str(k)+'nn_X_train.pkl'))
#joblib.dump(y_train, os.path.join(out_dir, 'RF_'+str(k)+'nn_y_train.pkl'))




# caculate and save score value, record tunning time

start_time = time.time()
rf = RandomForestClassifier(random_state=0)
clf = GridSearchCV(rf, parameters, verbose=3, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf)
print("---model training time: %s seconds ---" % (time.time() - start_time))
print("clf.best_score_: ", clf.best_score_)
joblib.dump(clf.best_score_, os.path.join(out_dir, 'RF_'+str(k)+'nn_nestimators'+str(n_estimators)+'_maxdepth'+str(max_depth)+'_score.pkl'))
#joblib.dump(clf, os.path.join(out_dir, 'RF_'+str(k)+'nn_model.pkl'))