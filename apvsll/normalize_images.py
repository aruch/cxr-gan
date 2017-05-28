import os
import h5py
from skimage import exposure

f = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "train.hdf5"),'r+')
images = f["images"]
images_norm = f.create_dataset('images_norm', images.shape, dtype='float32')
#images_norm = f["images_norm"]

for i in range(0, 75250, 50):
  print(i)
  currimages = images[i:i+50]
  for j in range(0, 50):
    images_norm[i + j]  = exposure.equalize_hist(currimages[j])

images_norm[75250:75286]  = exposure.equalize_hist(images[75250:75286])

f.close()
