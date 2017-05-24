import cPickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import h5py

N_TRAIN = 40000
N_TEST = 10000
def load_cxr(data_dir, subset='train', size=64):
    if subset=='train':
        h = h5py.File(os.path.join(data_dir, "train.hdf5"))
        trainx = np.array(h["images64"][:N_TRAIN])
        trainx = np.expand_dims(trainx, axis=1)
        trainy = np.array(h["label"][:N_TRAIN], dtype=np.int32)
        trainy[trainy==2] = 1

        return trainx, trainy
    elif subset=='test':
        h = h5py.File(os.path.join(data_dir, "train.hdf5"))
        trainx = np.array(h["images64"][N_TRAIN:(N_TRAIN+N_TEST)])
        trainx = np.expand_dims(trainx, axis=1)
        trainy = np.array(h["label"][N_TRAIN:(N_TRAIN+N_TEST)], dtype=np.int32)
        trainy[trainy==2] = 1

        return trainx, trainy
    else:
        raise NotImplementedError('subset should be either train or test')
