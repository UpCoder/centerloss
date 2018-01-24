# -*- coding=utf-8 -*-
import tensorflow as tf
from model import inference
import numpy as np
from reader import Reader
from plot import plot_scatter
from sklearn.decomposition import PCA


def test(images_tensor, labels_tensor, generator, parameters_path):
    '''
    测试模型的有效性
    :param images_tensor: 图像的tensor
    :param labels_tensor: label的tensor
    :param generator: 数据的生成器，可以通过for (images_batch, labels_batch) in generator:格式来获取数据
    :param parameters_path:模型保存的路径
    :return:
    '''
    center_feature, adversarial_feature = inference(images_tensor, True, True)
    saver = tf.train.Saver()
    featuremaps = []
    labels = []
    with tf.Session() as sess:
        full_path = tf.train.latest_checkpoint(parameters_path)
        print full_path
        saver.restore(sess, full_path)
        for (images_batch, labels_batch) in generator:
            print np.shape(images_batch)
            labels.extend(np.argmax(labels_batch, 1))
            featuremap_value = sess.run(center_feature,
                                        feed_dict={
                                            images_tensor: images_batch,
                                            labels_tensor: labels_batch
                                        })
            featuremaps.extend(featuremap_value)
    print np.shape(featuremaps), np.shape(labels)
    featuremaps = np.array(featuremaps)
    pca_obj = PCA(n_components=2)
    featuremaps = pca_obj.fit_transform(featuremaps)
    plot_scatter(featuremaps[:, 0], featuremaps[:, 1], labels, 10)

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data', training_epoch_num=1,
                    validation_epoch_num=1, testing_epoch_num=1)
    images_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_x')
    labels_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    test(images_tensor, labels_tensor, reader.train_generator, '/home/give/PycharmProjects/Reproduce/CenterLoss/unsupervised/parameters')