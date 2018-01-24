# Copyright 20170611 . All Rights Reserved.
# Prerequisites:
# Python 2.7
# gzip, subprocess, numpy
#
# ==============================================================================
"""Functions for downloading and uzip MNIST data."""
import gzip
import subprocess
import os
import numpy
from six.moves import urllib


def maybe_download(filename, data_dir, SOURCE_URL):
    """Download the data from Yann's website, unless it's already here."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def check_file(data_dir):
    if os.path.exists(data_dir):
        return True
    else:
        os.mkdir(data_dir)
        return False


def uzip_data(target_path):
    # uzip mnist data
    cmd = ['gzip', '-d', target_path]
    print('Unzip ', target_path)
    subprocess.call(cmd)


def read_data_sets(data_dir):
    if check_file(data_dir):
        print(data_dir)
        print('dir mnist already exist.')

        # delete the dir mnist
        cmd = ['rm', '-rf', data_dir]
        print('delete the dir', data_dir)
        subprocess.call(cmd)
        os.mkdir(data_dir)

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    data_keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
    for key in data_keys:
        if os.path.isfile(os.path.join(data_dir, key)):
            print("[warning...]", key, "already exist.")
        else:
            maybe_download(key, data_dir, SOURCE_URL)

    # uzip the mnist data.
    uziped_data_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte',
                        't10k-labels-idx1-ubyte']
    for key in uziped_data_keys:
        if os.path.isfile(os.path.join(data_dir, key)):
            print("[warning...]", key, "already exist.")
        else:
            target_path = os.path.join(data_dir, key)
            uzip_data(target_path)


if __name__ == '__main__':
    print("===== running - input_data() script =====")
    read_data_sets("./data")
    print("=============   =============")