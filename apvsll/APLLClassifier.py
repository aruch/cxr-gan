from sklearn.neighbors import KNeighborsClassifier
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle

TRAIN_DATA_SIZE = 1000
NEIGHBOR = 3
WEIGHT = "distance"

def prepDataSet(X, training, img_size):
  if training:
    datasize = TRAIN_DATA_SIZE
  else:
    datasize = X.shape[0]
  UnrollX = np.zeros((datasize, img_size * img_size))
  for i in range(0, datasize):
    UnrollX[i, : ] = X[i].ravel()
  return UnrollX

class APvLLClassifier(object):
  def __init__(self, img_size):
    pklfile = open('./train_labels.pkl', 'rb')
    labels = pickle.load(pklfile)
    pklfile.close()

    f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"))
    images = f["images" + str(img_size)]
    trainX= prepDataSet(images, True, img_size)
    self.img_size=img_size
    self.neigh = KNeighborsClassifier(n_neighbors=3, weights="distance")
    self.neigh.fit(trainX, labels)

  def classify(self, newX):
    newXUnroll = prepDataSet(newX, False, self.img_size)
    return self.neigh.predict(newXUnroll)

