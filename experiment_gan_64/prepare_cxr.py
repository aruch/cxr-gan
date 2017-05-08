from PIL import Image
from scipy.io import savemat
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import os
import h5py

def grab_image(f):
    f = h5py.File(f)
    img = f['data'][()].reshape(512, 512)
    img = (img * 255.).astype('uint8')
    f.close()
    return img

def resize(img):
    return np.array(Image.fromarray(img).resize((64, 64), Image.ANTIALIAS))

def collect_images(files):
    assert len(files) != 0, "Unable to find any files in source directory"
    img_buf = np.zeros((len(files), 64, 64, 1), dtype='uint8')

    for i, f in enumerate(tqdm(files)):
        img_buf[i] = resize(grab_image(f)).reshape(64, 64, 1)

    return img_buf

def main(args):
    dest = args.source + '-64x64.mat'
    print 'Opening from {:s} and saving to {:s}'.format(args.source, dest)

    files = glob(os.path.join(args.source, '*/*'))
    img_buf = collect_images(files)

    savemat(dest, {'images': img_buf})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=str,
                        default='/local-scratch/rshu15/nerdd/training',
                        help="Source for cxr")
    main(parser.parse_args())
