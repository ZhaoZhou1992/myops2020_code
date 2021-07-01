import math
import os
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Make sure we are not using all GPUs

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True # Try not to eat all the GPU memory
sess = tf.Session(config=config)

import keras
from keras import backend as K
K.set_session(sess)

import numpy as np
import cv2
import sys
import atriaseg as proc
import matplotlib
#from scipy.misc import imsave
import scipy.io as scio
from data_tools import *
import nibabel as nib
from scipy import ndimage
from imgaug import augmenters as iaa
import json
# TODO: Generate the data all at once
class DataGenerator(object):
        
    def data_generation(self):
        with open("list_id1.json", "r") as f:
            list_id = json.load(f)
        angle = 0
        name = 1
        for ID in list_id:
            for i in range(1,51):
                angle = i
                name = i
                rot = iaa.Affine(rotate=angle)
                #flip = iaa.Fliplr(1.0)
                print(ID)
                x_C0 = nib.load('/work/zz/MyoPS2020/crop_250*250/image/' + ID + '_C0.nii.gz').get_data()
                x_DE = nib.load('/work/zz/MyoPS2020/crop_250*250/image/' + ID + '_DE.nii.gz').get_data()
                x_T2 = nib.load('/work/zz/MyoPS2020/crop_250*250/image/' + ID + '_T2.nii.gz').get_data()
                gt = nib.load('/work/zz/MyoPS2020/crop_250*250/gt/' + ID + '_gd.nii.gz').get_data()
                for i in range(x_C0.shape[2]):
                    x_C0[:,:,i] = rot.augment_image(x_C0[:,:,i])
                    x_DE[:,:,i] = rot.augment_image(x_DE[:,:,i])
                    x_T2[:,:,i] = rot.augment_image(x_T2[:,:,i])
                '''
                x1 = np.copy(x).astype(np.float64)
                x1 -= ndimage.mean(x[x>0])
                x1 /= ndimage.standard_deviation(x[x>0])
                '''
                img_C0 = nib.Nifti1Image(x_C0, np.eye(4))
                img_DE = nib.Nifti1Image(x_DE, np.eye(4))
                img_T2 = nib.Nifti1Image(x_T2, np.eye(4))
                img_gt = nib.Nifti1Image(gt, np.eye(4))
                directory = "/work/zz/MyoPS2020/image_aug/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                img_C0.to_filename(os.path.join('/work/zz/MyoPS2020/crop_250*250/image/{}_{}_C0.nii.gz'.format(str(ID),str(name))))
                img_DE.to_filename(os.path.join('/work/zz/MyoPS2020/crop_250*250/image/{}_{}_DE.nii.gz'.format(str(ID),str(name))))
                img_T2.to_filename(os.path.join('/work/zz/MyoPS2020/crop_250*250/image/{}_{}_T2.nii.gz'.format(str(ID),str(name))))
                img_gt.to_filename(os.path.join('/work/zz/MyoPS2020/crop_250*250/gt/{}_{}_gd.nii.gz'.format(str(ID),str(name))))

                print(name)
                
                
if __name__ == "__main__":
    
    DataGenerator=DataGenerator()
    DataGenerator.data_generation()
