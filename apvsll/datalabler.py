#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import h5py
import pickle
import readchar
import os
import sys

# define '1' as ap, define '0' as ll, define 'q' as quit

class DataLabelerBackend(object):
  def __init__(self, labels=None):
    self.f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"))
    self.images = self.f["images"]
    if labels is None:
      self.labels=[]
    else:
      self.labels=labels
    self.current_index = len(labels)
    print(self.current_index)

  def nextPicture(self):
    self.current_index += 1
    return self.images[self.current_index]

  def prevPicture(self):
    self.current_index -= 1
    del(self.labels[-1])
    return self.images[self.current_index]

  def showPicture(self, img):
    plt.imshow(img, cmap='gray')    
    plt.show()

  def addLabel(self, label):
    self.labels.append(label)

  def pickle(self):
    output = open("train_labels.pkl", 'wb')
    pickle.dump(self.labels, output)
    output.close()


class DataLabler(object):
  def __init__(self, labels=None):
    if labels is not None:
      self.data = DataLabelerBackend(labels)
    else:
      self.data = DataLabelerBackend()


  def runInputLoop(self):
    cont = False
    img = self.data.images[self.data.current_index]
    while True:
      if cont == False:
        self.data.showPicture(img)
        plt.pause(.001)
      char = repr(readchar.readkey())
      if char == "'1'":
        cont = False
        img = self.data.nextPicture()
        self.data.addLabel(1);
        print("AP")
      elif char == "'0'":
        cont = False
        img = self.data.nextPicture()
        self.data.addLabel(0);
        print("LL")
      elif char == "'q'":
        cont = False
        self.data.pickle();
        print("Quit")
        return
      elif char == "'b'":
        cont = False
        img = self.data.prevPicture()
        print("Prev")
      else:
        cont = True
        continue

def main(argv=None):
  if argv is None:
    argv = sys.argv
  labels = None
  if len(sys.argv) == 2:
    pklfile = open(sys.argv[1], 'rb')
    labels = pickle.load(pklfile)
    pklfile.close()
  plt.ion()
  if labels is not None:
    user_input = DataLabler(labels)
  else:
    user_input = DataLabler()
  user_input.runInputLoop()
  return

if __name__ == "__main__":
    main()
