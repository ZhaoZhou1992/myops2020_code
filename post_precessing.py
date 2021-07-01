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
data_path_012 = '/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_012_/0_5'

data_012= glob.glob(data_path_012+"/pre/*")
dice_012_1=0
dice_012_2=0
for data_name in data_012:
    filter_size = [1,1,1]
    dataset = '0_5'        
    name=data_name[data_name.rindex("/")+1:-4]
    print(name)
    images_012=nib.load(data_path_012+"/pre/"+name+data_type).get_data()

    images_012_1=(images_012==1)
    images_012_2=(images_012==2)

    img = nib.Nifti1Image(images_012_1.astype('float64'), np.eye(4))
    directory = "result_segmentation_post/result_1_/{}/pre".format(dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_1_/{}/pre/'.format(dataset), '{}.nii'.format(name)))

    images_012_1=nib.load('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_1_/{}/pre/'.format(dataset)+name+data_type)
    if np.sum(images_012_1.get_data()>0):
    	images_012_1=image.largest_connected_component_img(images_012_1).get_data()
    	#images_012_1=ndimage.median_filter(images_012_1,size=filter_size)
    else:
    	images_012_1=images_012_1.get_data()


    img = nib.Nifti1Image(images_012_2.astype('float64'), np.eye(4))
    directory = "result_segmentation_post/result_2_/{}/pre".format(dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_2_/{}/pre/'.format(dataset), '{}.nii'.format(name)))

    images_012_2=nib.load('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_2_/{}/pre/'.format(dataset)+name+data_type)
    if np.sum(images_012_2.get_data()>0):
    	images_012_2=image.largest_connected_component_img(images_012_2).get_data()
    	#images_012_2=ndimage.median_filter(images_012_2,size=filter_size)
    else:
    	images_012_2=images_012_2.get_data()

    images_012_2[images_012_2==1]=2

    images_012_sum = images_012_1+images_012_2
    print(images_012_sum.shape)

    img = nib.Nifti1Image(images_012_sum.astype('float64'), np.eye(4))
    directory = "result_segmentation_post/result_sum_/{}/pre".format(dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation_post/result_sum_/{}/pre/'.format(dataset), '{}.nii'.format(name)))



