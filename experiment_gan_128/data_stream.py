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

class DataStream(Process):
    def __init__(self, data_q, file_q):
        Process.__init__(self)
        self.daemon = True
        self.data_q = data_q
        self.file_q = file_q
        self.start()

    def run(self):
        while True:
            files = self.file_q.get()
            images = read_files(files)
            self.data_q.put(images)

class Cxr(object):
    def __init__(self, bs=16, data_dir='/local-scratch/rshu15/nerdd/png', cache=10):
        self.bs = bs
        self.h5_files = glob(os.path.join(data_dir, 'training/*/*.png'))
        self.file_q = Queue()
        self.data_q = Queue()
        DataStream(self.data_q, self.file_q)

        for _ in xrange(cache):
            self.push(self.bs)

    def norm(self, x):
        return x - 0.5

    def denorm(self, x):
        return np.clip(x + 0.5, 0, 1)

    def push(self, bs):
        files = np.random.choice(self.h5_files, bs, replace=False)
        self.file_q.put(files)

    def next_batch(self, bs):
        self.push(bs)
        x = self.data_q.get()
        return x
