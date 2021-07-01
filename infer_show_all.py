# First setup tensorflow and keras, import the rest after
import os
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Make sure we are not using all GPUs
#os.environ['CUDA_VISIBLE_DEVICES'] = "2" 
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
import atriaseg as proc
import matplotlib
import cv2
#from scipy.misc import imsave
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from scipy import ndimage
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

#model = model.create_model(weights_file)
model_filename ='/lrde/home/zz/Heart/atriaseg2018/MyoPS2020/model_zz_epoch_49.h5'
model_json = model_segmentation.load_model_json(model_filename)
model = model_segmentation.load_model(model_filename)
optimizer = keras.optimizers.Adam(epsilon=0.002, amsgrad=True)
model.compile(optimizer, loss= ['binary_crossentropy','binary_crossentropy'],
    metrics=['accuracy',metrics.dice_coef_no_bg])

i = 0
for ID in list_id[20:25]:
    print(ID, i)
    crop_shape=(240,240)
    mri_C0 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_C0.nii.gz').get_data()
    mri_DE = nib.load('/work/zz/MyoPS2020/image/' + ID + '_DE.nii.gz').get_data()
    mri_T2 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_T2.nii.gz').get_data()

    norm_C0 = np.copy(mri_C0.astype(float)) 
    norm_C0 -= ndimage.mean(mri_C0)
    norm_C0 /= ndimage.standard_deviation(mri_C0)

    norm_DE = np.copy(mri_DE.astype(float)) 
    norm_DE -= ndimage.mean(mri_DE)
    norm_DE /= ndimage.standard_deviation(mri_DE)
    
    norm_T2 = np.copy(mri_T2.astype(float)) 
    norm_T2 -= ndimage.mean(mri_T2)
    norm_T2 /= ndimage.standard_deviation(mri_T2)

    X = np.zeros((norm_T2.shape[2],) + crop_shape+ (2,))
    X_1 = np.zeros((norm_T2.shape[2],) + crop_shape+ (1,))

    dataShape = norm_C0.shape
    w1 = int(np.ceil((dataShape[0]-crop_shape[0])/2.0))
    w2 = dataShape[0]-int(np.floor((dataShape[0]-crop_shape[0])/2.0))
    h1 = int(np.ceil((dataShape[1]-crop_shape[1])/2.0))
    h2 = dataShape[1]-int(np.floor((dataShape[1]-crop_shape[1])/2.0)) 
    for k in range(norm_T2.shape[2]):       
        X[k, :, :, 0] =norm_C0[w1:w2,h1:h2,k]
        #X[k, :, :, 1] =norm_DE[w1:w2,h1:h2,k]
        X[k, :, :, 1] =norm_T2[w1:w2,h1:h2,k]
        X_1[k, :, :, 0] =norm_T2[w1:w2,h1:h2,k]

    '''
    if "transpose_axis" in model_json:
        norm = np.transpose(norm, model_json["transpose_axis"])
    else:
        #transpose axis
        #  norm = np.transpose(norm, (2, 0, 1))
        #  gt = np.transpose(gt, (2, 0, 1))
        pass
    '''
    #norm = np.expand_dims(norm, axis=-1)
    #X = np.expand_dims(X, axis= 0)
    #X_1 = np.expand_dims(X_1, axis= 0)

    #layer_input = model.layers[0].input
    layer_input = model.get_layer('input_1').input
    layer_input1 = model.get_layer('input_2').input
    #layer_outputs = [layer.output for layer in model.layers[:18]]
    layer_outputs = model.get_layer('conv2d_48').output
    #layer_outputs = model.get_layer('reshape_1').output
    activation_model = keras.Model(inputs=[layer_input,layer_input1], outputs=[layer_outputs])
    activations = activation_model.predict([X,X_1])
    #activations = np.squeeze(activations, axis= -1)
    print(activations.shape)
    
    images_per_row = 1

    # Now let's display our feature maps
    for layer_activation in activations:
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        print(n_cols)

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[:, :,col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                print(channel_image.shape)
                channel_image = np.transpose(channel_image, tuple((1, 0)))
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

        # Display the grid
        plt.imsave('4_4/{}_{}.jpg'.format(str(ID),str(i)),display_grid)
        i+=1
    
