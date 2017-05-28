import os
import h5py
from APLLClassifier import APvLLClassifier

classifier = APvLLClassifier(64)
f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"),'r+')
images = f["images" + str(64)]

#isAP = f.create_dataset('isAP', (75286,), dtype='bool')

isAP = f["isAP"]

for i in range(35000, 75000, 500):
    print(i)
    isAP[i:i+500]  = classifier.classify(images[i:i+500])

isAP[75000:75286]  = classifier.classify(images[75000:75286])

f.close()
