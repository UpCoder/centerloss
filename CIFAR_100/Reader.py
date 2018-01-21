# -*-coding=utf-8 -*-
import os
from utils.Tools import split_array, indices_to_one_hot
from utils.Generator import GenerateBatch
import numpy as np
import cPickle


class Reader:
    def __init__(self, data_dir, batch_size, reshape_flag, is_fine=True, one_hot=True):
        '''
        加载cifar数据
        :param data_dir: Cifar100数据所在的文件夹，里面应该包括三个文件meta, train, test
        :param batch_size: 返回的一个batch的数据大小
        :param reshape_flag: 是否reshape，因为默认的是列向量
        :param is_fine: cifar100有两个label， 一个是fine一个是coarse，前者是小类别，共有100个，后者是大类别，共有20个
        '''
        self.file_pathes = []
        self.batch_size = batch_size
        self.training_file_path = os.path.join(data_dir, 'train')
        self.testing_file_path = os.path.join(data_dir, 'test')
        if (not os.path.exists(self.testing_file_path)) or (not os.path.exists(self.training_file_path)):
            print 'Please download the fully dataset'
            return
        # coarse_labels——一个范围在0 - 19
        # 的包含n个元素的列表, 对应图像的大类别
        # fine_labels——一个范围在0 - 99
        # 的包含n个元素的列表, 对应图像的小类别
        self.training_data, self.training_fine_labels, self.training_coarse_labels, self.training_filenames = self.generate_training_dataset()
        self.testing_data, self.testing_fine_labels, self.testing_coarse_labels, self.testing_filenames = self.generate_testing_dataset()
        if one_hot:
            self.training_fine_labels = indices_to_one_hot(self.training_fine_labels, 100)
            self.training_coarse_labels = indices_to_one_hot(self.training_coarse_labels, 20)
            self.testing_fine_labels = indices_to_one_hot(self.testing_fine_labels, 100)
            self.testing_coarse_labels = indices_to_one_hot(self.testing_coarse_labels, 20)

        split_rate = [0.8]
        data_splited = split_array(self.training_data, 2, split_rate)
        fine_labels_splited = split_array(self.training_fine_labels, 2, split_rate)
        coarse_labels_splited = split_array(self.training_coarse_labels, 2, split_rate)
        filenames_splited = split_array(self.training_filenames, 2, split_rate)
        self.training_data = data_splited[0]
        self.training_fine_labels = fine_labels_splited[0]
        self.training_coarse_labels = coarse_labels_splited[0]
        self.training_filenames = filenames_splited[0]
        self.val_data = data_splited[1]
        self.val_fine_labels = fine_labels_splited[1]
        self.val_coarse_labels = coarse_labels_splited[1]
        self.val_filenames = filenames_splited[1]
        if reshape_flag:
            self.training_data = self.reshape(np.array(self.training_data))
            self.val_data = self.reshape(np.array(self.val_data))
            self.testing_data = self.reshape(np.array(self.testing_data))
        if is_fine:
            self.train_generator = GenerateBatch(self.training_data, self.training_fine_labels, self.batch_size,
                                                 epoch_num=None).generate_next_batch()
            self.val_generator = GenerateBatch(self.val_data, self.val_fine_labels, self.batch_size,
                                               epoch_num=None).generate_next_batch()
            self.test_generator = GenerateBatch(self.testing_data, self.testing_fine_labels, self.batch_size,
                                                epoch_num=1).generate_next_batch()
        else:
            self.train_generator = GenerateBatch(self.training_data, self.training_coarse_labels, self.batch_size,
                                                 epoch_num=None).generate_next_batch()
            self.val_generator = GenerateBatch(self.val_data, self.val_coarse_labels, self.batch_size,
                                               epoch_num=None).generate_next_batch()
            self.test_generator = GenerateBatch(self.testing_data, self.testing_coarse_labels, self.batch_size,
                                                epoch_num=1).generate_next_batch()

    def generate_training_dataset(self):
        '''
        提取所有的训练数据
        :return:
        '''
        cur_data_batch = Reader.unpickle(self.training_file_path)
        return cur_data_batch['data'], cur_data_batch['fine_labels'], cur_data_batch['coarse_labels'], cur_data_batch[
            'filenames']

    def generate_testing_dataset(self):
        cur_data_batch = Reader.unpickle(self.testing_file_path)
        return cur_data_batch['data'], cur_data_batch['fine_labels'], cur_data_batch['coarse_labels'], cur_data_batch[
            'filenames']

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict

    @staticmethod
    def reshape(dataset):
        count = len(dataset)
        return dataset.reshape(count, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")