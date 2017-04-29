import h5py
import os
import numpy as np
import time
from PIL import Image
# from Queue import Queue
# from threading import Thread
from multiprocessing import Process, Queue

DATA_DIR = "/scratch/users/aruch/nerdd"

class DataLoader(Process):
    """Process for loading images from an hdf5 file"""
    def __init__(self, data_queue, idx_queue, img_size, images):
        Process.__init__(self)
        self.data_q = data_queue
        self.idx_q = idx_queue
        self.daemon = True
        self.images = images
        self.img_size = img_size
        self.start()

    def run(self):
        while True:
            idxs = self.idx_q.get(block=True)
            n = idxs.shape[0]

            batch = np.zeros((n,) + self.img_size)
                    
            for i in range(n):
                batch[i] = np.array(self.images[idxs[i]])

            self.data_q.put(batch, block=True)

class DataFeeder:
    def __init__(self, h5_name="train.hdf5", d=DATA_DIR, img_size=(64, 64),
                 cache_size=10, batch_size=16):
        """
        Args:
            h5_name (str): file name
            d (str): path name
            img_size (tuple): size of image to output in batches
            cache_size (int): number of batches to cache in memory
        """
        self.f = h5py.File(os.path.join(d, h5_name))
        if img_size==(64, 64):
            img_name = "images64"
        elif img_size==(128, 128):
            img_name = "images128"
        else:
            img_name = "images"

        images = self.f[img_name]

        self.n_images = images.shape[0]
        self.img_size = img_size
        self.cache_size = cache_size
        self.data_q = Queue(cache_size)
        self.idx_q = Queue(cache_size)
        self.batch_size = batch_size
        DataLoader(self.data_q, self.idx_q, img_size, images)

    def minibatches(self, random=True):
        pass

    def queue_next_indices(self):
        if self.img_n > self.n_images:
            return
        n = min(self.n_images-self.img_n, self.batch_size)
        idxs = self.idx[self.img_n:(self.img_n + n)]
        self.idx_q.put(idxs)
        self.img_n += n

    def prep_minibatches(self, random):
        """Prep indexes for minibatch creation

        Args:
            random (bool): randomly permute indexes or leave in order

        Returns:
            None
        """
        if random:
            self.idx = np.random.choice(self.n_images, self.n_images, replace=False)
        else: 
            self.idx = np.arange(self.n_images)
        self.img_n = 0
        for i in range(self.cache_size):
            self.queue_next_indices()

    def next_batch(self):
        """Get next batch

        Args:
             batch_size (int): number of images in batch

        Returns:
            batch: numpy array with batch_size number of images
        """
        self.queue_next_indices()
        return self.data_q.get(True, 4)

if __name__=="__main__":
    im_size = (64, 64)
    batch_size = 16 
    d = DataFeeder(img_size=im_size, batch_size=batch_size)
    d.prep_minibatches(random=True)

    print("Sleeping")
    time.sleep(5)
    print("Queue Full:", d.data_q.full())

    N = 10

    tic = time.time()
    for i in range(N):
        d.next_batch()

    toc = time.time()

    print((toc - tic)/N)

    print("Sleeping")
    time.sleep(1)
    print("Queue Full:", d.data_q.full())

    N = 100
    x = np.random.random((1500, 1500))

    tic = time.time()
    for i in range(N):
        y = np.dot(x,x)
    toc = time.time()

    print((toc - tic)/N)

    tic = time.time()
    for i in range(N):
        y = np.dot(x,x)
        d.next_batch()

    toc = time.time()

    print((toc - tic)/N)
