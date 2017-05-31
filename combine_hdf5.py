import h5py
import os
import numpy as np
import time
from PIL import Image

DATA_DIR = "/scratch/users/aruch/nerdd" 
def combine_datasets():
    names_and_attr = {}

    f = h5py.File(os.path.join(DATA_DIR, "training_partition-0.hdf5"))
    for n in f:
        names_and_attr[n] = [f[n].shape[1:], f[n].dtype]
    f.close()

    total_n = 0
    for i in range(0, 17):
        if i==3:
            continue
        f_name = "training_partition-{:}.hdf5".format(i)
        f = h5py.File(os.path.join(DATA_DIR, f_name))
        total_n += f["acc"].shape[0]
        f.close()

    agg_file = h5py.File(os.path.join(DATA_DIR, "train.hdf5"), "w")

    STEP = 100 

    for name in names_and_attr:
        print(name)
        if name=="images":
            size = (total_n,) + names_and_attr[name][0][:-1]
        else:
            size = (total_n,) + names_and_attr[name][0]
        dset = agg_file.create_dataset(name, size, dtype=names_and_attr[name][1])
        cur_n = 0
        for i in range(0, 17):
            if i==3:
                continue
            print(i)
            f_name = "training_partition-{:}.hdf5".format(i)
            f = h5py.File(os.path.join(DATA_DIR, f_name))
            n = f[name].shape[0]
            j = 0
            while j < n:
                step = min(n-j, STEP)
                if name=="images":
                    step_data = np.array(f[name][j:(j+step)]).squeeze()
                else:
                    step_data = f[name][j:(j+step)]
                dset[(cur_n+j):(cur_n+j+step)] = step_data
                j += step
            cur_n += j
            print(cur_n)
            f.close()

    agg_file.close()

def add_downsample_images(file_path):
    agg_file = h5py.File(file_path)

    size_64 = (64, 64)
    size_128 = (128, 128)
    size_256 = (256, 256)

    n = agg_file["images_norm"].shape[0]

    if "images_norm64" in agg_file:
        dset64 = agg_file["images_norm64"]
    else:
        dset64 = agg_file.create_dataset("images_norm64", (n,) + size_64, dtype="float32")
    if "images_norm128" in agg_file:
        dset128 = agg_file["images_norm128"]
    else:
        dset128 = agg_file.create_dataset("images_norm128", (n,) + size_128, dtype="float32")
    if "images_norm256" in agg_file:
        dset256 = agg_file["images_norm256"]
    else:
        dset256 = agg_file.create_dataset("images_norm256", (n,) + size_256, dtype="float32")

    tic = time.time()
    for i in range(n):
        if i % 500 == 0:
            toc = time.time()
            print i
            print toc - tic
            tic = time.time()
        img = Image.fromarray(agg_file["images_norm"][i], "F")
        img64 = img.resize(size_64, Image.ANTIALIAS)
        img128 = img.resize(size_128, Image.ANTIALIAS)
        img256 = img.resize(size_256, Image.ANTIALIAS)
        dset64[i] = np.array(img64)
        dset128[i] = np.array(img128)
        dset256[i] = np.array(img256)

    agg_file.close()
