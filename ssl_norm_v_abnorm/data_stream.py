import h5py
import os
import numpy as np
import time
from PIL import Image
# from Queue import Queue
# from threading import Thread
from multiprocessing import Process, Queue

DATA_DIR = "/scratch/users/aruch/nerdd"

class DataStream:
    def __init__(self, h5_name="train.hdf5", d=DATA_DIR, img_size=64,
                 n_img_per_seg=2000, batch_size=16):
        """
        Args:
            h5_name (str): file name
            d (str): path name
            img_size (tuple): size of image to output in batches
            cache_size (int): number of batches to cache in memory
        """
        self.f = h5py.File(os.path.join(d, h5_name))
        if img_size==64:
            img_name = "images64"
        elif img_size==128:
            img_name = "images128"
        else:
            img_name = "images_norm"

        self.images = self.f[img_name]
        self.labels = self.f["label"]

        self.n_images = self.images.shape[0]
        self.n_batches = int(np.ceil(self.n_images / batch_size))
        self.batch_per_seg = int(np.round(n_img_per_seg / batch_size))
        self.image_per_seg =  self.batch_per_seg * batch_size
        self.n_seg = int(np.ceil(self.n_images / self.image_per_seg))
        self.img_size = img_size
        self.batch_size = batch_size

    def load_next_segment(self, random=True):
        if self.cur_seg >= self.n_seg:
            return
        seg_start = self.cur_seg * self.image_per_seg
        seg_end = min(self.n_images, seg_start + self.image_per_seg) 
        self.image_cache = self.images[seg_start:seg_end]
        self.label_cache = self.labels[seg_start:seg_end]
        self.label_cache[self.label_cache==2] = 1
        self.idxs = np.random.permutation(seg_end-seg_start)
        self.cur_batch = 0
        self.batch_in_seg = int(np.ceil(float(seg_end-seg_start)/self.batch_size))
        self.cur_seg += 1

    def prep_minibatches(self, random=True):
        """Prep indexes for minibatch creation

        Args:
            random (bool): randomly permute indexes or leave in order

        Returns:
            None
        """
        if random:
            self.seg_idx = np.random.permutation(self.n_seg)
        else: 
            self.seg_idx = np.arange(self.n_seg)

        self.cur_seg = 0
        self.load_next_segment(random)

    def next_batch(self, timeout=1, dims=4):
        """Get next batch

        Args:
             batch_size (int): number of images in batch

        Returns:
            batch: numpy array with batch_size number of images
        """
        if self.cur_batch >= self.batch_in_seg:
            self.load_next_segment()

        batch_start = self.cur_batch * self.batch_size
        batch_end = min(self.idxs.shape[0], batch_start + self.batch_size)
        ret_x = self.image_cache[self.idxs[batch_start:batch_end]]
        ret_y = np.array(self.label_cache[self.idxs[batch_start:batch_end]], dtype=np.int32)
        self.cur_batch += 1

        if dims==4:
            ret_x = np.expand_dims(ret_x, axis=1)
            
        return ret_x, ret_y

if __name__=="__main__":
    im_size = 512
    batch_size = 16
    n_img_per_seg = 2000

    d = DataStream(img_size=im_size, batch_size=batch_size, n_img_per_seg=n_img_per_seg)

    print("Batch Size: {:}".format(batch_size))
    print("Image Size: {:} x {:}".format(*im_size))

    N = 40000/batch_size
    
    tic = time.time()
    d.prep_minibatches(random=True)

    for i in range(N):
        z = d.next_batch()

    toc = time.time()

    print("Takes {:.4f}s per batchload".format((toc - tic)/N))
    print(toc-tic)
