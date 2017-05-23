import os
import h5py
from APLLClassifier import APvLLClassifier

classifier = APvLLClassifier(64)
f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"))
images = f["images" + str(64)]


newlabels = classifier.classify(images[1000:1010])
print(newlabels)

