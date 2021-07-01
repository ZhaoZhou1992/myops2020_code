# First setup tensorflow and keras, import the rest after
import os
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "2" # Make sure we are not using all GPUs

import tensorflow as tf

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True # Try not to eat all the GPU memory
sess = tf.Session(config=config)

import keras
from keras import backend as K
K.set_session(sess)

import json
import sys
import keras
import keras.models
import progress.bar
import nibabel as nib
import metrics
import timeit
from data_tools import *
import model_segmentation
import os
from scipy import ndimage
from math import floor
from timeit import default_timer as timer

#Parse the arguments first
def parse_args():
    from docopt import docopt
    from ast import literal_eval

    from schema import Schema, Use, And, Or

    usage = """
Usage: infer.py -h
       infer.py [--training_set=<path>] <model_path> <output_location>

Options:
    -h --help                  show this
    --training_set=<path>      the training dataset location
    <model_path>               the path to the model weights
    <output_location>          the path to the output folder

    """

    args = docopt(usage)
    schema = Schema({
        '--help': False,
        '--training_set': Or(None, And(Use(str), os.path.isdir)),
        '<output_location>': str,
        '<model_path>': str,
        })
    try:
        args = schema.validate(args)
    except Exception as e:
        print(args)
        print(e)
        import sys
        sys.exit(1)

    #args['--transpose'] = literal_eval(args['--transpose'])

    return args

#args = parse_args()


with open("list_id1.json", "r") as f:
    list_id = json.load(f)

training_folder = '/work/ykhoudli/2018_AtriaSeg/Training Set/'
if training_folder is None:
    training_folder = '/work/ykhoudli/2018_AtriaSeg/Training Set/'

#model_filename ='/lrde/home/zz/Heart/atriaseg2018/ISBI2020_WHS_1/step_without_FOA/012_weight_fold_16_20/model_epoch_7.h5'metrics.SSIM_LOSS_myops
model_filename ='/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/model_epoch_1.h5'
model_json = model_segmentation.load_model_json(model_filename)
model = model_segmentation.load_model(model_filename)
optimizer = keras.optimizers.Adam(epsilon=0.01, amsgrad=True)
model.compile(optimizer, loss=[metrics.SSIM_LOSS,metrics.SSIM_LOSS_myops],
    metrics=['accuracy', metrics.dice_coef_no_bg])

i = 0
full_time = 0
noload_time = 0
predtime = 0
w1,w2,w2_restore,h1,h2,h2_restore=0,0,0,0,0,0

for ID in list_id[0:5]:
    axis = '012_'
    dataset = '0_5'
    print(ID, i)
    crop_shape=(240,240)    
    sload = timer()
    base = training_folder+ID+'/'
    mri_C0 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_C0.nii.gz').get_data()
    mri_DE = nib.load('/work/zz/MyoPS2020/image/' + ID + '_DE.nii.gz').get_data()
    mri_T2 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_T2.nii.gz').get_data()

    gt = nib.load('/work/zz/MyoPS2020/gt/' + ID + '_gd.nii.gz').get_data()
    start = timer()
    dataShape = mri_C0.shape
    w1 = int(np.ceil((dataShape[0]-crop_shape[0])/2.0))
    w2 = dataShape[0]-int(np.floor((dataShape[0]-crop_shape[0])/2.0))
    w2_restore = int(np.floor((dataShape[0]-crop_shape[0])/2.0))
    h1 = int(np.ceil((dataShape[1]-crop_shape[1])/2.0))
    h2 = dataShape[1]-int(np.floor((dataShape[1]-crop_shape[1])/2.0))
    h2_restore = int(np.floor((dataShape[1]-crop_shape[1])/2.0))
    ''' 
    norm_C0 = np.copy(mri_C0[w1:w2,h1:h2,:].astype(float)) 
    norm_C0 -= ndimage.mean(mri_C0[w1:w2,h1:h2,:])
    norm_C0 /= ndimage.standard_deviation(mri_C0[w1:w2,h1:h2,:])

    norm_DE = np.copy(mri_DE[w1:w2,h1:h2,:].astype(float)) 
    norm_DE -= ndimage.mean(mri_DE[w1:w2,h1:h2,:])
    norm_DE /= ndimage.standard_deviation(mri_DE[w1:w2,h1:h2,:])
    
    norm_T2 = np.copy(mri_T2[w1:w2,h1:h2,:].astype(float)) 
    norm_T2 -= ndimage.mean(mri_T2[w1:w2,h1:h2,:])
    norm_T2 /= ndimage.standard_deviation(mri_T2[w1:w2,h1:h2,:])
    '''

    X = np.zeros((norm_T2.shape[2],) + crop_shape+ (3,))
    X_1 = np.zeros((norm_T2.shape[2],) + crop_shape+ (1,))

    y = np.zeros(crop_shape+(norm_T2.shape[2],))
    y_myo = np.zeros(crop_shape+(norm_T2.shape[2],))
    

    for k in range(norm_T2.shape[2]): 
        norm_C0[:,:,k] = np.copy(mri_C0[w1:w2,h1:h2,k].astype(float)) 
        norm_C0[:,:,k] -= ndimage.mean(mri_C0[w1:w2,h1:h2,k])
        norm_C0[:,:,k] /= ndimage.standard_deviation(mri_C0[w1:w2,h1:h2,k])

        norm_DE[:,:,k] = np.copy(mri_DE[w1:w2,h1:h2,k].astype(float)) 
        norm_DE[:,:,k] -= ndimage.mean(mri_DE[w1:w2,h1:h2,k])
        norm_DE[:,:,k] /= ndimage.standard_deviation(mri_DE[w1:w2,h1:h2,k])
    
        norm_T2[:,:,k] = np.copy(mri_T2[w1:w2,h1:h2,k].astype(float)) 
        norm_T2[:,:,k] -= ndimage.mean(mri_T2[w1:w2,h1:h2,k])
        norm_T2[:,:,k] /= ndimage.standard_deviation(mri_T2[w1:w2,h1:h2,k])


        X[k, :, :, 0] =norm_C0[:,:,k]
        X[k, :, :, 1] =norm_DE[:,:,k]
        X[k, :, :, 2] =norm_T2[:,:,k]
        X_1[k, :, :, 0] =norm_DE[:,:,k]
    y[:,:,:] = gt[w1:w2,h1:h2,:]
    y[y==200] = 0
    y[y==500] = 0
    y[y==600] = 0
    y[y==1220] = 1
    y[y==2221] = 2

    y_myo[:,:,:] = gt[w1:w2,h1:h2,:]
    y_myo[y_myo==200] = 1
    y_myo[y_myo==500] = 0
    y_myo[y_myo==600] = 0
    y_myo[y_myo==1220] = 1
    y_myo[y_myo==2221] = 1


    sp = timer()
    pred = model.predict([X,X_1],verbose=False,batch_size=1)
    ep = timer()

    
    if "transpose_axis" in model_json:
        inverted_transpose = np.argsort(model_json["transpose_axis"])
        print(inverted_transpose)
        gt = np.transpose(gt, inverted_transpose)
        pred_1 = np.transpose(pred[1], tuple((1, 2, 0)) + (3,))
        pred_2 = np.transpose(pred[0], tuple((1, 2, 0)) + (3,))
        #pred = np.transpose(pred, tuple(inverted_transpose) + (3,))    
    #  gt = keras.utils.to_categorical(gt, num_classes=2)
    #  print(model.evaluate(X, gt))

    # Maximize and remove one-hot encoding
    proba = pred
    pred_1 = np.argmax(pred_1, 3)
    pred_2 = np.argmax(pred_2, 3)
    pred_1_restore = np.pad(pred_1,[[w1,w2_restore],[h1,h2_restore],[0,0]],'constant')
    pred_2_restore = np.pad(pred_2,[[w1,w2_restore],[h1,h2_restore],[0,0]],'constant')
    #pred = np.round(resize_image(pred.astype(float),(0.65,0.65,0.65),(spacing[2],spacing[0],spacing[1])))
    end = timer()
    full_time += end - sload
    noload_time += end - start
    predtime += ep -sp
    pred_1 = pred_1.astype('float64')
    img = nib.Nifti1Image(pred_1, np.eye(4))
    directory = "result_segmentation/result_{}/{}/pre".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/pre/'.format(axis,dataset), '{}.nii'.format(ID)))
    pred_2 = pred_2.astype('float64')
    img = nib.Nifti1Image(pred_2, np.eye(4))
    directory = "result_segmentation/result_{}/{}/pre_myo".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/pre_myo/'.format(axis,dataset), '{}.nii'.format(ID)))


    img = nib.Nifti1Image(gt.astype('float64'), np.eye(4))
    directory = "result_segmentation/result_{}/{}/gt".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/gt/'.format(axis,dataset), '{}_gt.nii'.format(ID)))
    img = nib.Nifti1Image(y.astype('float64'), np.eye(4))
    directory = "result_segmentation/result_{}/{}/crop_gt".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/crop_gt/'.format(axis,dataset), '{}_gt.nii'.format(ID)))
    img = nib.Nifti1Image(y_myo.astype('float64'), np.eye(4))
    directory = "result_segmentation/result_{}/{}/crop_gt_myo".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/crop_gt_myo/'.format(axis,dataset), '{}_gt.nii'.format(ID)))

    pred_1_restore = pred_1_restore.astype('float64')
    img = nib.Nifti1Image(pred_1_restore, np.eye(4))
    directory = "result_segmentation/result_{}/{}/pre_restore".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/pre_restore/'.format(axis,dataset), '{}.nii'.format(ID)))
    pred_2_restore = pred_2_restore.astype('float64')
    img = nib.Nifti1Image(pred_2_restore, np.eye(4))
    directory = "result_segmentation/result_{}/{}/pre_myo_restore".format(axis,dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    img.to_filename(os.path.join('/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/result_segmentation/result_{}/{}/pre_myo_restore/'.format(axis,dataset), '{}.nii'.format(ID)))
    i += 1
    
print(full_time / 2)
print(noload_time / 2)
print(predtime / 2)
    
