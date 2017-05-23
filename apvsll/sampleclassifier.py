import os
import h5py
from APLLClassifier import APvLLClassifier

IMG_SIZE = 64 

classifier = APvLLClassifier(IMG_SIZE)
f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"))
images = f["images" + str(IMG_SIZE)]


newlabels = classifier.classify(images[1000:1010])
print(newlabels)

