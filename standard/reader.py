# -*- coding=utf-8 -*-
import numpy as np
import struct
import os
from utils.Tools import shuffle_image_label, split_array, indices_to_one_hot


def conver2npy_image(data_path, save_path):
    '''
    将下载得到图片进行解析，转化成numpy的格式 shape: (n,28,28)
    :param data_path: 下载得到Image文件的路径
    :param save_path: 保存numpy的路径
    :return:
    '''
    filename = data_path
    binfile = open(filename, 'rb')
    buf = binfile.read()
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index) # 读取数据头
    index += struct.calcsize('>IIII')
    print numImages, numRows, numColumns
    images_arr = np.zeros([numImages, numRows, numColumns])
    for image_index in range(numImages):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        images_arr[image_index, :, :] = np.reshape(np.array(im), [numRows, numColumns])
    np.save(save_path, images_arr)
    return images_arr


def convert2npy_label(data_path, save_path):
    '''
        将下载得到标签进行解析，转化成numpy的格式 shape:(n)
        :param data_path: 下载得到Label文件的路径
        :param save_path: 保存numpy的路径
        :return:
        '''
    filename = data_path
    binfile = open(filename, 'rb')
    buf = binfile.read()
    index = 0
    magic, labels = struct.unpack_from('>II', buf, index)  # 读取数据头
    index += struct.calcsize('>II')
    print labels
    labels_arr = np.zeros([labels])
    for label_index in range(labels):
        label = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
        labels_arr[label_index] = label
    np.save(save_path, labels_arr)
    return labels_arr

def read_npys(data_path):
    train_image = np.load(os.path.join(data_path, 'train_images.npy'))
    train_label = np.load(os.path.join(data_path, 'train_labels.npy'))
    train_image = np.expand_dims(train_image, 3)
    train_label = indices_to_one_hot(train_label, 10)
    test_image = np.load(os.path.join(data_path, 'test_images.npy'))
    test_label = np.load(os.path.join(data_path, 'test_labels.npy'))
    test_image = np.expand_dims(test_image, 3)
    test_label = indices_to_one_hot(test_label, 10)
    return train_image, train_label, test_image, test_label


class Reader:
    def __init__(self, data_path, batch_size=100):
        self.data_dir = data_path
        self.batch_size = batch_size
        self.train_image, self.train_label, self.test_image, self.test_label = read_npys(self.data_dir)
        self.train_image, self.train_label = shuffle_image_label(self.train_image, self.train_label)
        self.test_image, self.test_label = shuffle_image_label(self.test_image, self.test_label)
        split_rate = [0.8]
        splited_images = split_array(self.train_image, num=2, rate=split_rate)
        splited_labels = split_array(self.train_label, num=2, rate=split_rate)
        self.train_image = splited_images[0]
        self.train_label = splited_labels[0]
        self.val_image = splited_images[1]
        self.val_label = splited_labels[1]
        print np.shape(self.train_image), np.shape(self.val_image), np.shape(self.test_image)
        print np.shape(self.train_label), np.shape(self.val_label), np.shape(self.test_label)
        self.train_generator = GenerateBatch(self.train_image, self.train_label, self.batch_size,
                                             epoch_num=None).generate_next_batch()
        self.val_generator = GenerateBatch(self.val_image, self.val_label, self.batch_size,
                                           epoch_num=None).generate_next_batch()
        self.test_generator = GenerateBatch(self.test_image, self.test_label, self.batch_size,
                                            epoch_num=1).generate_next_batch()


class GenerateBatch:
    def __init__(self, dataset, label, batch_size, epoch_num=None):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.start = 0
        self.epoch_num = epoch_num

    def generate_next_batch(self):
        if self.epoch_num is not None:
            for i in range(self.epoch_num):
                while self.start < len(self.dataset):
                    cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                    cur_label_batch = self.label[self.start: self.start + self.batch_size]
                    self.start += self.batch_size
                    cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                    yield cur_image_batch, cur_label_batch
        else:
            while True:
                cur_image_batch = self.dataset[self.start: self.start + self.batch_size]
                cur_label_batch = self.label[self.start: self.start + self.batch_size]
                self.start = (self.start + self.batch_size) % len(self.dataset)
                cur_image_batch = np.asarray(cur_image_batch, np.float32) / 255.0
                yield cur_image_batch, cur_label_batch

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    for (image_batch, label_batch) in reader.test_generator:
        image_batch = np.array(image_batch)
        print np.shape(image_batch), np.shape(label_batch)
        from PIL import Image
        img = Image.fromarray(np.asarray(image_batch[0, :, :, :]).squeeze())
        print label_batch[0]
        img = img.resize([100,100])
        img.show()
        input('%d')
