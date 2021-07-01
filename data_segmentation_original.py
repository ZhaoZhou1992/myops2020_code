import math
from math import floor
import keras
import numpy as np
import sys
import logging
from data_tools import *
from scipy import ndimage
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize

# TODO: Generate the data all at once
class DataGenerator(keras.utils.Sequence):
    "Generate the data on the fly to be used by keras"
    def __init__(self, list_IDs, batch_size=1, dim=(88, 576, 576), n_channels=3,crop_shape=(240,240),
            n_classes=2,n_classes_myo=3, shuffle=True, transpose_axis = (0, 1, 2)):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.crop_shape = crop_shape
        self.n_classes = n_classes
        self.n_classes_myo = n_classes_myo
        self.shuffle = shuffle
        self.cut_dim = (self.dim[0],) + tuple(math.ceil(self.dim[i] * 3 / 5) for i in [1, 2])
        self.n_slice = self.cut_dim[transpose_axis[0]]
        self.transpose_axis = tuple(transpose_axis)
        self.load_id = None
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

      tmp = []
      for i in self.indexes:
          gt=nib.load('/work/zz/MyoPS2020/gt_aug/' + self.list_IDs[i] + '_gd.nii.gz').get_data()
          #spacing=sitk.ReadImage('/work/zz/MyoPS2020/gt/' + self.list_IDs[i] + '_gd.nii.gz').GetSpacing()
          #self.n_slice = np.round(resize_image(image.astype(float),(spacing[2],spacing[0],spacing[1]),(0.65,0.65,0.65))).shape[0]
          self.n_slice = gt.shape[2]
          slice_indexes = np.arange(self.n_slice)
          np.random.shuffle(slice_indexes)
          tmp += zip([i] * self.n_slice, slice_indexes)

      self.indexes = tmp

    # TODO: Make volume caching thread local
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        '''
        dim = tuple(self.cut_dim[i] for i in self.transpose_axis[1:])
        X = np.empty((self.batch_size,) + dim + (self.n_channels,))
        x_C0 = np.empty(dim+ (20,), dtype=float )
        x_DE = np.empty(dim+ (20,), dtype=float )
        x_T2 = np.empty(dim+ (20,), dtype=float )
        '''

        w1,w2,h1,h2=0,0,0,0

        # Generate data
        for i, (ID, k) in enumerate(list_IDs_temp):
          # Store sample
          #if self.load_id != ID:
          logging.info("opening sample {}".format(ID))

          self.x_C0 = nib.load('/work/zz/MyoPS2020/image_aug/' + ID + '_C0.nii.gz').get_data()
          self.x_DE = nib.load('/work/zz/MyoPS2020/image_aug/' + ID + '_DE.nii.gz').get_data()
          self.x_T2 = nib.load('/work/zz/MyoPS2020/image_aug/' + ID + '_T2.nii.gz').get_data()
          #spacing=sitk.ReadImage('/work/zz/ISBI2020_WHS/Ground_truth/' + ID + '_label.nii.gz').GetSpacing()
          #self.x = np.round(resize_image(self.x.astype(float),(spacing[2],spacing[0],spacing[1]),(0.65,0.65,0.65),order=1))
          '''
          X = np.zeros((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]) + (self.n_channels,))
          X1 = np.zeros((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]) + (self.n_channels,))
          X2 = np.zeros((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]) + (self.n_channels,))
          '''
          x_C0 = np.empty((self.batch_size,)+(self.x_C0.shape[0],self.x_C0.shape[1],self.x_C0.shape[2]))
          x_DE = np.empty((self.batch_size,)+(self.x_DE.shape[0],self.x_DE.shape[1],self.x_DE.shape[2]))
          x_T2 = np.empty((self.batch_size,)+(self.x_T2.shape[0],self.x_T2.shape[1],self.x_T2.shape[2]))
          '''
          y = np.empty((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]))
          y_C0 = np.empty((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]))
          y_DE = np.empty((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]))
          y_T2 = np.empty((self.batch_size,) + (self.x_T2.shape[0],self.x_T2.shape[1]))
          '''


          
          X = np.zeros((self.batch_size,) + self.crop_shape+ (self.n_channels,))
          #X1 = np.zeros((self.batch_size,) + self.crop_shape + (self.n_channels,))
          #X2 = np.zeros((self.batch_size,) + self.crop_shape + (self.n_channels,))
          y = np.empty((self.batch_size,) + self.crop_shape)
          y_1 = np.empty((self.batch_size,) + self.crop_shape)
          X_DE = np.zeros((self.batch_size,) + self.crop_shape+ (1,))
          '''
          y_C0 = np.empty((self.batch_size,) + self.crop_shape)
          y_DE = np.empty((self.batch_size,) + self.crop_shape)
          y_T2 = np.empty((self.batch_size,) + self.crop_shape)
          
          logging.debug("minmax {} - {}".format(self.x.min(), self.x.max()))

          hamming_window = np.dot(np.hamming(self.x.shape[1])[:, None],np.hanning(self.x.shape[2])[None, :])
          sigma = np.mean([self.x.shape[1],self.x.shape[2]])
          [rs, cs] = np.mgrid[(- floor(self.x.shape[1]/2)): (self.x.shape[1]-floor(self.x.shape[1]/2)) , (-floor(self.x.shape[2]/2)):(self.x.shape[2]-floor(self.x.shape[2]/2))]
          dist = rs**2+cs**2
          window = hamming_window*np.exp(-0.5 / (sigma**2) *(dist))
          window = window/np.sum(window)

          for j in range(self.x.shape[0]):
            self.x[ j, :, :]=np.multiply(self.x[ j, :, :],window)
            self.x[ j, :, :]=(self.x[ j, :, :]-np.min(self.x[ j, :, :]))/(np.max(self.x[ j, :, :])-np.min(self.x[ j, :, :])+0.1)*255
          '''         
          x_C0 = np.copy(self.x_C0.astype(float))
          x_C0 -= ndimage.mean(self.x_C0)
          x_C0 /= ndimage.standard_deviation(self.x_C0)

          x_DE = np.copy(self.x_DE.astype(float))
          x_DE -= ndimage.mean(self.x_DE)
          x_DE /= ndimage.standard_deviation(self.x_DE)
          
          x_T2 = np.copy(self.x_T2.astype(float))
          x_T2 -= ndimage.mean(self.x_T2)
          x_T2 /= ndimage.standard_deviation(self.x_T2)


          self.y = nib.load('/work/zz/MyoPS2020/gt_aug/' + ID + '_gd.nii.gz').get_data()
          y_initial = np.zeros((self.batch_size,) +(self.y.shape[0],self.y.shape[1],self.y.shape[2]))
          y_1_initial = np.zeros((self.batch_size,) + (self.y.shape[0],self.y.shape[1],self.y.shape[2]))
          y_initial = np.copy(self.y)
          y_1_initial = np.copy(self.y)
          y_initial[y_initial==200] = 1
          y_initial[y_initial==500] = 0
          y_initial[y_initial==600] = 0
          y_initial[y_initial==1220] = 1
          y_initial[y_initial==2221] = 1

          
          y_1_initial[y_1_initial==200] = 0
          y_1_initial[y_1_initial==500] = 0
          y_1_initial[y_1_initial==600] = 0
          y_1_initial[y_1_initial==1220] = 1
          y_1_initial[y_1_initial==2221] = 2

          #self.y = np.round(resize_image(self.y.astype(float),(spacing[2],spacing[0],spacing[1]),(0.65,0.65,0.65)))
          self.load_id = ID
          dataShape = self.x_C0.shape
          
          w1 = int(np.ceil((dataShape[0]-self.crop_shape[0])/2.0))
          w2 = dataShape[0]-int(np.floor((dataShape[0]-self.crop_shape[0])/2.0))
          h1 = int(np.ceil((dataShape[1]-self.crop_shape[1])/2.0))
          h2 = dataShape[1]-int(np.floor((dataShape[1]-self.crop_shape[1])/2.0)) 
          '''
          X[i, :, :, 0] =x_C0[w1:w2,h1:h2,k]
          X1[i, :, :, 0] =x_DE[w1:w2,h1:h2,k]
          X2[i, :, :, 0] =x_T2[w1:w2,h1:h2,k]
          '''
          X[i, :, :, 0] = x_C0[w1:w2,h1:h2, k]
          X[i, :, :, 1] = x_DE[w1:w2,h1:h2, k]
          X[i, :, :, 2] = x_T2[w1:w2,h1:h2, k]
          X_DE[i, :, :, 0] = x_DE[w1:w2,h1:h2, k]
          '''
          X[i, :, :, 0] = x_C0[:,:, k]
          X[i, :, :, 1] = x_DE[:,:, k]
          X[i, :, :, 2] = x_T2[:,:, k]
	  
          # Store class
          y[i,] = self.y[:,:,k]
          '''
          y[i,] = y_initial[w1:w2,h1:h2,k]
          y_1[i,] = y_1_initial[w1:w2,h1:h2,k]

          '''
          y_C0[i,] = self.y[w1:w2,h1:h2,k]
          y_DE[i,] = self.y[w1:w2,h1:h2,k]
          y_T2[i,] = self.y[w1:w2,h1:h2,k]
          '''
        #return X,X1,X2, keras.utils.to_categorical(y, num_classes=self.n_classes), keras.utils.to_categorical(y_C0, num_classes=self.n_classes), keras.utils.to_categorical(y_DE, num_classes=self.n_classes), keras.utils.to_categorical(y_T2, num_classes=self.n_classes)return X,X_T2, keras.utils.to_categorical(y, num_classes=self.n_classes), keras.utils.to_categorical(y_1, num_classes=self.n_classes_myo)
        return X,X_DE, keras.utils.to_categorical(y, num_classes=self.n_classes), keras.utils.to_categorical(y_1, num_classes=self.n_classes_myo)
        #return X_DE, keras.utils.to_categorical(y_1, num_classes=self.n_classes_myo)
    def __len__(self):
        'Denotes the number of batches per epoch'
        num=0
        for i in range(len(self.list_IDs)):
            gt=nib.load('/work/zz/MyoPS2020/gt_aug/' + self.list_IDs[i] + '_gd.nii.gz').get_data()
            #spacing=sitk.ReadImage('/work/zz/ISBI2020_WHS/Ground_truth/' + self.list_IDs[i] + '_label.nii.gz').GetSpacing()
            #num +=np.round(resize_image(image_gt.astype(float),(spacing[2],spacing[0],spacing[1]),(0.65,0.65,0.65))).shape[0]
            num +=gt.shape[2]
        return int(np.floor(num / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(self.list_IDs[k[0]], k[1]) for k in indexes]

        # Generate data
        X,X1, y,y1 = self.__data_generation(list_IDs_temp)

        return [X,X1],[y,y1]

