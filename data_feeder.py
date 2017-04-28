import h5py
import os
import numpy as np
import time
from PIL import Image

DATA_DIR = "/scratch/users/aruch/NeRDD"

class DataFeeder:
    def __init__(self, h5_name="train.hdf5", d=DATA_DIR, img_size=(64, 64)):
        """
        Args:
            h5_name (str): file name
            d (str): path name
            img_size (tuple): size of image to output in batches
        """
        self.f = h5py.File(os.path.join(d, h5_name))
        self.n_images = self.f["images"].shape[0]
        self.orig_img_size = self.f["images"].shape[1:]
        self.img_size = img_size
        self.idx = np.arange(self.n_images)

    def minibatches(self, random=True, batch_size=16):
        pass

    def prep_minibatches(self, random):
        """Prep indexes for minibatch creation

        Args:
            random (bool): randomly permute indexes or leave in order

        Returns:
            None
        """
        self.f["images"].size
        if random:
            self.idx = np.random.choice(self.n_images, self.n_images, replace=False)
        else: 
            self.idx = np.arange(self.n_images)
        self.img_n = 0

    def next_batch(self, batch_size):
        """Get next batch

        Args:
             batch_size (int): number of images in batch

        Returns:
            batch: numpy array with batch_size number of images
        """
        n = min(self.n_images-self.img_n, batch_size)
        idxs = self.idx[self.img_n:(self.img_n + n)]
        batch = np.zeros((n,) + self.img_size)
                    
        for i in range(n):
            img = Image.fromarray(self.f["images"][idxs[i]], "F")
            img = img.resize(self.img_size, Image.ANTIALIAS)
            batch[i] = np.array(img)

        self.img_n += n

        return batch

if __name__=="__main__":
    d = DataFeeder()
    d.prep_minibatches(random=False)

    N = 100
    bs = 64

    tic = time.time()
    for i in range(N):
        d.next_batch(bs)
    toc = time.time()

    print((toc - tic)/N)
