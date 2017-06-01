import os
import h5py

fold = h5py.File(os.path.join('/scratch/users/aruch/nerdd', "val_sub.hdf5"),'r+')
fnew = h5py.File("val_sub.hdf5", "a")

images_norm64 = fnew.create_dataset('images_norm64', data = fold["images_norm64"][:])
isAP = fnew.create_dataset('isAP', data=fold["isAP"][:])
label = fnew.create_dataset('label', data=fold["label"][:])

fold.close()
fnew.close()
