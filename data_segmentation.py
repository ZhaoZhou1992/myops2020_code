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
    def __init__(self, list_IDs, batch_size=1, dim=(88, 576, 576), n_channels=2,crop_shape=(240,240),
            n_classes=2,n_classes_myo=2, shuffle=True, transpose_axis = (0, 1, 2)):
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
          gt=nib.load('/work/zz/MyoPS2020/gt/' + self.list_IDs[i] + '_gd.nii.gz').get_data()
          self.n_slice = gt.shape[2]
          slice_indexes = np.arange(self.n_slice)
          np.random.shuffle(slice_indexes)
          tmp += zip([i] * self.n_slice, slice_indexes)

      self.indexes = tmp

    # TODO: Make volume caching thread local
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,) + self.crop_shape+ (self.n_channels,))

        w1,w2,h1,h2,mu,sigma=0,0,0,0,0,0
        y = np.zeros((self.batch_size,) + self.crop_shape+(1,))
        y_1 = np.zeros((self.batch_size,) + self.crop_shape+ (1,))
        y_2 = np.zeros((self.batch_size,) + self.crop_shape+ (1,))
        X_DE = np.zeros((self.batch_size,) + self.crop_shape+ (1,))
        # Generate data
        for i, (ID, k) in enumerate(list_IDs_temp):
          # Store sample
          #if self.load_id != ID:
          logging.info("opening sample {}".format(ID))

          self.x_C0 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_C0.nii.gz').get_data()
          self.x_DE = nib.load('/work/zz/MyoPS2020/image/' + ID + '_DE.nii.gz').get_data()
          self.x_T2 = nib.load('/work/zz/MyoPS2020/image/' + ID + '_T2.nii.gz').get_data()
          
          dataShape = self.x_C0.shape
          w1 = int(np.ceil((dataShape[0]-self.crop_shape[0])/2.0))
          w2 = dataShape[0]-int(np.floor((dataShape[0]-self.crop_shape[0])/2.0))
          h1 = int(np.ceil((dataShape[1]-self.crop_shape[1])/2.0))
          h2 = dataShape[1]-int(np.floor((dataShape[1]-self.crop_shape[1])/2.0))
          
          mu_C0 = ndimage.mean(self.x_C0)
          sigma_C0 = ndimage.standard_deviation(self.x_C0)
          if sigma_C0>0:
            #self.x_C0 = (self.x_C0-mu_C0)/(sigma_C0)  
            self.x_C0 = np.clip((self.x_C0-mu_C0)/(sigma_C0),-1,1)
          mu_DE = ndimage.mean(self.x_DE)
          sigma_DE = ndimage.standard_deviation(self.x_DE)
          if sigma_DE>0:
            #self.x_DE = (self.x_DE-mu_DE)/(sigma_DE) 
            self.x_DE = np.clip((self.x_DE-mu_DE)/(sigma_DE),-1,1)
          mu_T2 = ndimage.mean(self.x_T2)
          sigma_T2 = ndimage.standard_deviation(self.x_T2)
          if sigma_T2>0:
            #self.x_T2 = (self.x_T2-mu_T2)/(sigma_T2) 
            self.x_T2 = np.clip((self.x_T2-mu_T2)/(sigma_T2),-1,1)          


          self.y = nib.load('/work/zz/MyoPS2020/gt/' + ID + '_gd.nii.gz').get_data()
          y_initial = np.zeros((self.batch_size,) +(self.y.shape[0],self.y.shape[1],self.y.shape[2]))
          y_1_initial = np.zeros((self.batch_size,) + (self.y.shape[0],self.y.shape[1],self.y.shape[2]))
          y_2_initial = np.zeros((self.batch_size,) + (self.y.shape[0],self.y.shape[1],self.y.shape[2]))
          y_initial = np.copy(self.y)
          y_1_initial = np.copy(self.y)
          y_2_initial = np.copy(self.y)
          
          y_initial[y_initial==200] = 1
          y_initial[y_initial==500] = 0
          y_initial[y_initial==600] = 0
          y_initial[y_initial==1220] = 1
          y_initial[y_initial==2221] = 1
          
          
          y_1_initial[y_1_initial==200] = 0 
          y_1_initial[y_1_initial==500] = 0
          y_1_initial[y_1_initial==600] = 0
          y_1_initial[y_1_initial==1220] = 1
          y_1_initial[y_1_initial==2221] = 1
          
          y_2_initial[y_2_initial==200] = 0 
          y_2_initial[y_2_initial==500] = 0
          y_2_initial[y_2_initial==600] = 0
          y_2_initial[y_2_initial==1220] = 0
          y_2_initial[y_2_initial==2221] = 1
          

          self.load_id = ID

          X[i, :, :, 0] = self.x_C0[w1:w2,h1:h2,k]
          #X[i, :, :, 1] = self.x_DE[w1:w2,h1:h2,k]
          X[i, :, :, 1] = self.x_T2[w1:w2,h1:h2,k]

          #X_DE[i, :, :, 0] = self.x_C0[w1:w2,h1:h2,k]
          X_DE[i, :, :, 0] = self.x_DE[w1:w2,h1:h2,k]
          #X_DE[i, :, :, 2] = self.x_T2[w1:w2,h1:h2,k]

          y[i,:, :,0] = y_initial[w1:w2,h1:h2,k]
          y_1[i,:, :,0] = y_1_initial[w1:w2,h1:h2,k]
          y_2[i,:, :,0] = y_2_initial[w1:w2,h1:h2,k]

        #return X,X_DE, keras.utils.to_categorical(y_1, num_classes=self.n_classes_myo), keras.utils.to_categorical(y_2, num_classes=self.n_classes_myo)
        return X,X,X,X_DE, y_1,y_2
        #return X, y_1, y_2
        #return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        num=0
        for i in range(len(self.list_IDs)):
            gt=nib.load('/work/zz/MyoPS2020/gt/' + self.list_IDs[i] + '_gd.nii.gz').get_data()
            num +=gt.shape[2]
        return int(np.floor(num / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(self.list_IDs[k[0]], k[1]) for k in indexes]

        # Generate data
        X,X1,X2,X3,y1, y2= self.__data_generation(list_IDs_temp)

        return [X,X1,X2,X3],[y1,y2]

