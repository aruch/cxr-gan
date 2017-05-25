from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import h5py

def unzip(basename, zip_dir, dest_dir):
    zip_path_orig = os.path.join(zip_dir, basename)
    zip_path_new = os.path.join(dest_dir, basename)
    if not os.path.exists(zip_path_new):
        print "Copying {:s} to {:s}".format(zip_path_orig, dest_dir)
        os.system('cp {:s} {:s}'.format(zip_path_orig, dest_dir))
    if not os.path.exists(zip_path_new.rstrip('.zip')):
        print "Unzipping {:s}".format(zip_path_new)
        os.system('unzip {:s} -d {:s}'.format(zip_path_new, dest_dir))

def grab_numpy_image(h5_file):
    f = h5py.File(h5_file)
    img = f['data'][()].reshape(512, 512)
    img = (img * 255.).astype('uint8')
    f.close()
    return img

def resize_as_Image(img, size):
    return Image.fromarray(img).resize((size, size), Image.ANTIALIAS)

def replace_file(h5_file):
    png_file = h5_file.replace('.h5', '.png')
    if not os.path.exists(png_file):
        image = resize_as_Image(grab_numpy_image(h5_file), 128)
        image.save(png_file)
    os.remove(h5_file)

def mass_replace_set(name):
    print "Replacing {:s} set".format(name)
    h5_files = glob(os.path.join(dest_dir, '{:s}/*/*.h5'.format(name)))
    for h5_file in tqdm(h5_files):
        replace_file(h5_file)

def main(zip_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        unzip('training.zip', zip_dir, dest_dir)
        unzip('validation.zip', zip_dir, dest_dir)

    mass_replace_set('validation')
    mass_replace_set('training')

if __name__ == '__main__':
    zip_dir = '/scratch/PI/langlotz/nerdd/orig'
    dest_dir = '/local-scratch/rshu15/nerdd/png'
    main(zip_dir, dest_dir)
