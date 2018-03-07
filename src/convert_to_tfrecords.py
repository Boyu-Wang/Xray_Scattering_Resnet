'''
The file convert both image and labels into tfrecords files
Label is parsed from the corresponding xml file
'''
import os
import numpy as np
import xml.etree.ElementTree as ET
import struct
import scipy.io as sio
from glob import glob
import multiprocessing
from joblib import Parallel, delayed

raw_data_path = '../data/raw'
tfrecords_data_path = '../data/tfrecords'
tag_file = '../data/17tags_meta.txt'
tags_meta = []
with open(tag_file) as f:
    tags_meta = f.read().lower().split('\n')
NUM_TAGS = len(tags_meta)
# tags_meta is a list of tuples, each element is a tuples, like (0, 'bcc')
tags_meta = zip(np.arange(0, NUM_TAGS), tags_meta)
TRAIN_RATIO = 0.8
IMAGESIZE = 256

# data_file_name could be like ../../data/SyntheticScatteringData/014267be_varied_sm/00000001.mat
# ../../data/SyntheticScatteringData/014267be_varied_sm/analysis/results/00000001.xml
# return a label vector of size [num_of_tags]
def parse_label(data_file_name):
    data_path = data_file_name.rsplit('/',1)
    label_file_name = data_path[0] + '/analysis/results/' + data_path[1].split('.')[0] + '.xml'
    label_vector = np.zeros([NUM_TAGS])

    if os.path.exists(label_file_name):
        root = ET.parse(label_file_name).getroot()
        for result in root[0]:
            attribute = result.attrib.get('name')
            attribute = attribute.rsplit('.', 1)
            # only care about high level tags
            attribute = attribute[1].split(':')[0]
            # check if that attribute is with the all_tag_file
            for i in range(NUM_TAGS):
                if tags_meta[i][1] == attribute:
                    label_vector[tags_meta[i][0]] = 1
    else:
        print ('%s does not exist!' %label_file_name)

    flag = bool(sum(label_vector))
    return label_vector, flag

# helper function for binary data
# return a string of binary files about all files contain in that directory
# store label first, then image
# they are both encoded in float64 (double) format
def _binaryize_one_dir(dir):
    file_names = os.listdir(dir)
    string_binary = ''
    for data_file in file_names:
        if os.path.isfile(os.path.join(dir, data_file)):
            label, flag = parse_label(os.path.join(dir, data_file))
            if not flag:
                print(os.path.join(dir, data_file))
            else:
                label = label.astype('int16')
                label = list(label)
                label_byte = struct.pack('h'*len(label), *label)
                string_binary += label_byte
                image = sio.loadmat(os.path.join(dir, data_file))
                # the shape of image is 256*256
                image = image['detector_image']
                image = np.reshape(image, [-1])
                # take the log
                image = np.log(image) / np.log(1.0414)
                image[np.isinf(image)] = 0
                image = image.astype('int16')
                image = list(image)
                image_byte = struct.pack('h'*len(image), *image)
                string_binary += image_byte

    return string_binary


def _get_one_binary_file(dirs, save_name, i):
    print('processing %s_%d.bin' % (save_name, i))
    with open(os.path.join(tfrecords_data_path, '%s_%d.bin' % (save_name, i)), 'wb') as f:
        for dir_list_i in dirs:
            f.write(_binaryize_one_dir(dir_list_i))


def processFile():
    # dirs = os.listdir(DATA_PATH)
    dirs = glob(raw_data_path+'/*')
    num_dirs_per_bin = 50
    idx = np.random.permutation(len(dirs))
    num_bin_file = int(np.ceil(len(dirs) / num_dirs_per_bin))
    num_train_bin_file = int(TRAIN_RATIO * num_bin_file)
    num_val_bin_file = num_bin_file - num_train_bin_file

    # get training data
    train_dirs = []
    for i in range(num_train_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range(i*num_dirs_per_bin, (i+1)*num_dirs_per_bin)]
        train_dirs.append(tmp_dir)

    # get val data
    val_dirs = []
    for i in range(num_val_bin_file):
        tmp_dir = [dirs[idx[j]] for j in range((i+num_train_bin_file)*num_dirs_per_bin,  (i+1+num_train_bin_file)*num_dirs_per_bin)]
        val_dirs.append(tmp_dir)

    if not os.path.exists(tfrecords_data_path):
        os.mkdir(tfrecords_data_path)
    print(num_train_bin_file)
    print(num_val_bin_file)

    num_cores = multiprocessing.cpu_count()/2
    Parallel(n_jobs=num_cores)(
        delayed(_get_one_binary_file)(train_dirs[i], 'train_batch', i) for i in range(num_train_bin_file))

    Parallel(n_jobs=num_cores)(
        delayed(_get_one_binary_file)(val_dirs[i], 'val_batch', i) for i in range(num_val_bin_file))


def main():
    processFile()


if __name__ == "__main__":
    main()