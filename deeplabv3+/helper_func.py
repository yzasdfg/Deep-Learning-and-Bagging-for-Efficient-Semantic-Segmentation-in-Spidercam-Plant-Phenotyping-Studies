import os,random, logging
import glob
import pandas as pd
import numpy as np 
import cv2, math, sys, random, time

#from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from skimage.feature import graycomatrix, graycoprops

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix



def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_global_determinism(seed=42, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return

    logging.warning("*******************************************************************************")
    logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
    logging.warning("*******************************************************************************")

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'



H, W = 256, 256



def plot_loss_accuracy(history, save=False, filename='history.png'):
    
    plt.figure(figsize = (15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    if save: plt.savefig(filename, dpi=1000, bbox_inches='tight')


def intersection_union(y_pred, y_true, num_classes):
    '''
    y_pred and y_true: for one obs
    This function compute the intersection and union for the specific obs
    '''
    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    #assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)
    intersection, union=[], []
    y_pred = np.array([ y_pred==i for i in range(num_classes) ])#.transpose(1,2,3,0)

    #mask1=mask1[tf.newaxis,...]
    y_true = np.array([y_true==i for i in range(num_classes) ])#.transpose(1,2,3,0)
    for i in range(num_classes):
        intersection.append(np.sum(np.logical_and(y_pred[i], y_true[i])))
        union.append(np.sum(np.logical_or(y_pred[i], y_true[i])))
        #union = [np.nan if x == 0 else x for x in union]
        
        #iou=(np.array(intersection)+0.001) / (np.array(union)+0.001)

    #print('intersection, union: ', intersection, union)
    assert len(intersection) == num_classes, 'intersection should have length equal to the num_classes, instead are {}'.format(len(intersection))
    assert len(union) == num_classes, 'union should have length equal to the num_classes, instead are {}'.format(len(union))
    return np.array(intersection), np.array(union)


def iou_dice(y_pred, y_true, num_classes):
    '''
    y_pred and y_true: for one obs
    This function compute the iou for the specific obs
    '''

    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    #assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)
    intersection, union=[], []
    y_pred = np.array([ y_pred==i for i in range(num_classes) ])#.transpose(1,2,3,0)

    #mask1=mask1[tf.newaxis,...]
    y_true = np.array([y_true==i for i in range(num_classes) ])#.transpose(1,2,3,0)
    for i in range(num_classes):
        intersection.append(np.sum(np.logical_and(y_pred[i], y_true[i])))
        union.append(np.sum(np.logical_or(y_pred[i], y_true[i])))
        #union = [np.nan if x == 0 else x for x in union]
        
        iou=(np.array(intersection)+0.01) / (np.array(union)+0.01)
    #print('intersection: ', intersection)
    #print('union：', union)
    return iou    



def trainGenerator(batch_size,image_dataframe, mask_dataframe, x_col,y_col,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, **aug_dict)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_dataframe(
        image_dataframe,
        x_col='filename', y_col='class',
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_dataframe(
        mask_dataframe,
        x_col='filename', y_col='class',
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        #img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def train_val_split(image_list, mask_list, train_val_size, seed):
    '''
    This function split train/val from whole dataset. train:val=4:1. If train_percent =0.8, 
    80% of the whole data will be assigned to train, the rest of the data will be all assigned to val.
    '''
    random.seed(seed)
    assert len(image_list) == len(mask_list), \
    'Input masks should be same length, instead are {}, {}'.format(len(image_list), len(mask_list))
    #train_indx=random.sample(range(len(image_list)), int(len(image_list)*train_percent))
    train_val_indx=random.sample(range(len(image_list)), train_val_size)

    train_val_image_list= list( image_list[i] for i in train_val_indx )
    train_val_mask_list= list( mask_list[i] for i in train_val_indx )
    

    train_image_list, val_image_list, train_mask_list, val_mask_list=\
    train_test_split(train_val_image_list, train_val_mask_list, test_size=0.2, random_state=seed)

    unseleclted_image_list=list(image_list[i] for i in range(len(image_list)) if i not in train_val_indx)

    #rest_indx=[range(len(image_list))[i] for i in range(len(image_list)) if i not in train_indx]
    #val_indx=random.sample(rest_indx, len(train_indx)//4)
    #val_image_list = list( image_list[i] for i in val_indx )
    #val_mask_list = list( mask_list[i] for i in val_indx )
    return train_image_list, val_image_list, train_mask_list, val_mask_list, unseleclted_image_list


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

