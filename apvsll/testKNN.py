from sklearn.neighbors import KNeighborsClassifier
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle

IMAGE_SIZE = 64
USE_FULL = True

def unison_shuffled_copies(a, b):
  p = np.random.permutation(len(b))
  return a[p], b[p]

def prepDataSet(X, y):
  print("-------Preprocessing-------")
  UnrollX = np.zeros((1000, IMAGE_SIZE * IMAGE_SIZE))
  for i in range(0, 1000):
    UnrollX[i, : ] = X[i].ravel()
  
  #X, y = unison_shuffled_copies(UnrollX, np.array(y))
  
  X, y = UnrollX, np.array(y)
  trainX = X[0:800, :]
  valX = X[800:900, :]
  testX = X[900:1000, :]
  trainy = y[0:800]
  valy = y[800:900]
  testy = y[900:1000]
  return trainX, trainy, valX, valy, testX, testy

def trainAndTestKNN(trainX, trainy, valX, valy, testX, testy):
  bestAccuracy, bestNeighbor, bestWeight = -1, -1, None
  n_neighbors = [1, 3, 5]
  weights = ["uniform", "distance"]

  print("-------Begin Training-------")
  for neighbor in n_neighbors:
    for weight in weights:
      print("-------Training " + str(neighbor) + " nn with " + weight + "weight---------------")
      neigh = KNeighborsClassifier(n_neighbors=neighbor, weights=weight)
      neigh.fit(trainX, trainy)
      valaccuracy = neigh.score(valX, valy)
      print(valaccuracy)
      if valaccuracy >= bestAccuracy:
        if neighbor >= bestNeighbor:
          bestAccuracy = valaccuracy
          bestNeighbor = neighbor
          bestWeight = weight

  print("-------Testing " + str(bestNeighbor) + " nn with " + bestWeight + "weight---------------")
  neigh = KNeighborsClassifier(n_neighbors=bestNeighbor, weights=bestWeight)
  neigh.fit(trainX, trainy)
  print(neigh.score(testX, testy))

def main(argv=None):
  if argv is None:
    argv = sys.argv
  
  pklfile = open('./train_labels.pkl', 'rb')
  labels = pickle.load(pklfile)
  pklfile.close()

  f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"))
  images = f["images" + str(IMAGE_SIZE)]
  trainX, trainy, valX, valy, testX, testy = prepDataSet(images, labels)
  trainAndTestKNN(trainX, trainy, valX, valy, testX, testy)

if __name__ == "__main__":
  main()


