import os
import glob
import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage 
from skimage import morphology
from nilearn import image
#import atriaseg as proc
from data_tools import *
#from sklearn.metrics import f1_score

data_type = ".nii"
data_path_012 = '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_sum_/0_5'
data_path_012_gt = '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_012_/0_5'
data_012= glob.glob(data_path_012+"/pre/*")
dice_012_1=0
dice_012_2=0
dice_012_3=0
dice_012_4=0
dice_012_5=0
dice_012_1_myo=0
for data_name in data_012:        
    name=data_name[data_name.rindex("/")+0:-4]
    print(name)
    images_012=nib.load(data_path_012+"/pre"+name+data_type).get_data()   
    images_gt=nib.load(data_path_012_gt+"/crop_gt"+name+"_gt"+data_type).get_data()

    print(images_gt.shape)


    images_012_1 = (images_012 == 1)
    images_gt_1 = (images_gt==1)  
    dice012_1 = 2.0*(np.sum(np.logical_and(images_012_1, images_gt_1)))/(np.sum(images_012_1) + np.sum(images_gt_1))
    dice_012_1+=dice012_1
    images_012_2 = (images_012 == 2)
    images_gt_2 = (images_gt==2)  
    dice012_2 = 2.0*(np.sum(np.logical_and(images_012_2, images_gt_2)))/(np.sum(images_012_2) + np.sum(images_gt_2))   
    dice_012_2+=dice012_2
    

print('dice_012_1:'+str(dice_012_1/len(data_012))+'\n')
print('dice_012_2:'+str(dice_012_2/len(data_012))+'\n')
'''
print('dice_012_3:'+str(dice_012_3/len(data_012))+'\n')
print('dice_012_4:'+str(dice_012_4/len(data_012))+'\n')
print('dice_012_5:'+str(dice_012_5/len(data_012))+'\n')
'''
