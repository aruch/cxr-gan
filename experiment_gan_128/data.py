import numpy as np
from glob import glob
from PIL import Image
import os
from multiprocessing import Process, Queue
import time

def read_files(files):
    images = np.stack([np.array(Image.open(f)) for f in files])
    # reshape and reformat
    images = images.reshape(-1, 128, 128, 1)
    images = images.astype('float32') / 255. - 0.5
    return images

class Cxr(object):
    def __init__(self, bs=16, data_dir='/local-scratch/rshu15/nerdd/png', cache=10):
        self.h5_files = glob(os.path.join(data_dir, 'training/*/*.png'))

    def norm(self, x):
        return x - 0.5

    def denorm(self, x):
        return np.clip(x + 0.5, 0, 1)

    def next_batch(self, bs):
        files = np.random.choice(self.h5_files, bs, replace=False)
        x = read_files(files)
        return x
