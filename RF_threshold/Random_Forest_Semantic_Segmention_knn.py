#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
After hyperparameter tunning, finalized the model
predict batches of images in parallel
have to retrain the RF everytime because, not able to save model due to computing capacity
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
from sklearn.ensemble import RandomForestClassifier
import sys
import time
import random, tensorflow as tf

from itertools import product
from skimage.feature import graycomatrix, graycoprops
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import seaborn as sns, pandas as pd


from skimage.feature import graycomatrix, graycoprops
def list_glcm(gray, d, levels=256, arg_list=['contrast', 'dissimilarity','homogeneity', 'correlation', 'ASM']): 
        # get a gray level co-occurrence matrix (GLCM)
    # parameters：the matrix of gray image，distance，direction，gray level，symmetric or not，standarzation or not
    #levels: The input image should contain integers in [0, levels-1],
    glcm = graycomatrix(gray, d,  [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                        levels=256, symmetric = True, normed = True)
            
    
    mean=[]
    glcm_range=[]
    #获取共生矩阵的统计值.
    for prop in arg_list:
        
        temp = graycoprops(glcm, prop)
        mean.append(np.mean(temp, axis=1))
        glcm_range.append(np.nanmax(temp, axis=1)-np.nanmin(temp, axis=1))
    
    mean = [item for items in mean for item in items]
    return mean, glcm_range

def cm_analysis(y_true, y_pred, filename = 'confusion matrix.png', save_file = False, ymap=None, figsize=(10,10)):

    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        #labels = [ymap[yi] for yi in labels]
    labels = unique_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]
            if i == j:
                
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.2f%%\n%d/%d' % (p, c, s)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True label'
    cm.columns.name = 'Predicted label'
    
    fig, ax = plt.subplots(figsize=figsize)
  
    ax.set_title('confusion matrix')
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap = "Blues")
    if save_file == True:   
        plt.savefig(filename, bbox_inches ="tight", dpi=1000) 
    #plt.show()

##################################################################################################################################################################################
input_path  = r'/work/yzstat/yzhan/Maize_images/'


rs=42
k= int(sys.argv[1])
n_points= int(sys.argv[2])
test_image_index= int(sys.argv[3])
interval= int(sys.argv[4])
out_dir_prefix = sys.argv[5]



print('k nearest neighbor: ', k)
print('points per training image: ', n_points)
print('test image index: ', test_image_index)
print('interval: ', interval)
print('out_dir_prefix: ', out_dir_prefix )

out_dir=out_dir_prefix + str(k)+'nn_points_per_image_'+str(n_points)

pred_dir='/lustre/work/yzstat/yzhan/Maize_images_RF/multiclass_prediction/'+str(k)+'nn_points_per_image_'+str(n_points)+'/RF_'+str(k)+'nn'

# create output folder
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

start_index_list=np.arange(0, len(test_mask_patch_list), interval)
print('start_index_list: ', start_index_list)
start_index=start_index_list[test_image_index]
print('start_index: ', start_index)


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
# construct training samples. R, G, B, GLCM values at k neighbors and n_points per image
seed(rs)
random.seed(rs)

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

start_time = time.time()
clf = RandomForestClassifier(n_estimators=200, max_depth=50, random_state=0)
clf.fit(X_train, y_train)
#print(clf)
print("---model training time: %s seconds ---" % (time.time() - start_time))
#joblib.dump(clf, os.path.join(out_dir, 'RF_'+str(k)+'nn_model.pkl'))

# In[227]:


#k=1, 0.7643097643097643
#k=2, 0.7769360269360269
#k=3  0.7710437710437711
#k=4 0.7904040404040404
#k=5 0.7651515151515151

# In[234]:

y_pred= clf.predict(X_train)
cm_analysis(y_train, y_pred, os.path.join(out_dir, 'RF_'+str(k)+'nn_train_cm.png'), save_file=True )





#X_test, y_test=[], []
#mask=cv2.copyMakeBorder(cv2.imread(selected_mask_patch_list[test_image_index], 0),k,k,k,k,cv2.BORDER_REPLICATE)
#img=cv2.copyMakeBorder(cv2.imread(selected_image_patch_list[test_image_index]),k,k,k,k,cv2.BORDER_REPLICATE)
#gray_img=cv2.copyMakeBorder(cv2.imread(selected_image_patch_list[test_image_index], 0),k,k,k,k,cv2.BORDER_REPLICATE)

#filename=os.path.basename(selected_image_patch_list[test_image_index])

#point = random.sample(all_points, n_points)
#point = randint(k, 512-k, (n_points, 2)) # sample int from [low, high)


# prediction
score=[]
for i in range(len(test_image_patch_list))[start_index:start_index+interval]:     

    X_test, y_test=[], []
    
    mask=cv2.copyMakeBorder(cv2.imread(test_mask_patch_list[i], 0),k,k,k,k,cv2.BORDER_REPLICATE)  
    img=cv2.copyMakeBorder(cv2.imread(test_image_patch_list[i]),k,k,k,k,cv2.BORDER_REPLICATE)  
    gray_img=cv2.copyMakeBorder(cv2.imread(test_image_patch_list[i], 0),k,k,k,k,cv2.BORDER_REPLICATE)
    #mask=cv2.imread(mask_patch_list[i], 0)
    #img=cv2.imread(image_patch_list[i])
    #gray_img= cv2.imread(image_patch_list[i], 0)
    filename=os.path.basename(test_image_patch_list[i])

    for ap in all_points:
        #RGB features
        neighbor_img=[img[ap[0]+c[0], ap[1]+c[1], :] for c in coordinate]
        flat_list = [item for sublist in neighbor_img for item in sublist]
        #print(len(neighbor_img))
        #print(neighbor_img)
        
        #if flat_list in X_test:
            #if y_test[X_test.index(flat_list)]==mask[point[p][0], point[p][1]]: continue
        

        #GLCM features
        sub_gray_img=gray_img[ap[0]+coordinate[0][0]: ap[0]+coordinate[-1][0],  ap[1]+coordinate[0][1]: ap[1]+coordinate[-1][1]]
        #bins = np.linspace(0, 255, 64)        # gray level:64
        #compress_gray = np.digitize(img, bins)
        #gray = np.uint8(compress_gray) 

        mean, _=list_glcm(sub_gray_img, d=[k])# data type of the image£ºuint8
        
        #combine the two features
        flat_list.extend(mean)
            
        X_test.append(flat_list)
        y_test.append(mask[ap[0], ap[1]])



#joblib.dump(X_train, os.path.join(out_dir, 'RF_'+str(k)+'nn_X_train.pkl'))
#joblib.dump(y_train, os.path.join(out_dir, 'RF_'+str(k)+'nn_y_train.pkl'))
#joblib.dump(X_test, os.path.join(out_dir, 'RF_'+str(k)+'nn_X_test.pkl'))
#joblib.dump(y_test, os.path.join(out_dir, 'RF_'+str(k)+'nn_y_test.pkl'))

# In[233]:

#X_train=joblib.load('RF_'+str(k)+'nn_X_train.pkl')
#y_train=joblib.load('RF_'+str(k)+'nn_y_train.pkl')
#X_test=joblib.load('RF_'+str(k)+'nn_X_test.pkl')
#y_test=joblib.load('RF_'+str(k)+'nn_y_test.pkl')


    score.append(clf.score(X_test, y_test))

    # In[221]:

    start_time = time.time()
    y_pred= clf.predict(X_test)
    y_pred_save=y_pred.reshape(H, W)
    tf.keras.preprocessing.image.save_img(os.path.join(pred_dir, filename), y_pred_save[..., tf.newaxis], data_format=None, file_format='png', scale=False)

    #joblib.dump(y_pred, os.path.join(pred_dir, 'RF_'+str(k)+'nn_y_pred_'+str(test_image_index)+'.pkl'))


    print("--- model prediction time: %s seconds ---" % (time.time() - start_time))
joblib.dump(score, os.path.join(out_dir, 'RF_'+str(k)+'nn_score_' +str(start_index) + '.pkl'))
