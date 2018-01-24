# -*- coding=utf-8 -*-
import tensorflow as tf
from model import inference
from reader import Reader
import numpy as np
import os
category_num = 10
_lambda = 0.01
_alpha = 0.5
step_value = 0


def update_centers(centers, data, labels, category_num):
    '''
    更新质心
    :param centers: 所有质心的数组 type: numpy.array shape: [category_num, 2]
    :param data: 一个batch数据的features
    :param labels: 一个batch数据的label one hot编码格式
    :param category_num: 一个有多少种label，在这里就是10
    :return: 更新之后的centers
    '''
    global step_value
    centers = np.array(centers)
    # if step_value % 100 == 0:
    #     print centers
    # step_value += 1
    data = np.array(data)
    labels = np.array(np.argmax(labels, 1))
    centers_batch = centers[labels]
    diff = centers_batch - data
    for i in range(category_num):
        cur_diff = diff[labels == i]
        cur_diff = cur_diff / (1.0 + 1.0 * len(cur_diff))
        cur_diff = cur_diff * _alpha
        for j in range(len(cur_diff)):
            centers[i, :] -= cur_diff[j, :]
    # 到这里完成了让类内距离更近的操作

    # 下面尝试加入类间距离更大的操作

    return centers


def train(images_tensor, label_tensor, iterator_num, reader, summary_path='./log', restore=None):
    '''
    训练
    :param images_tensor: 图像数据的tensor
    :param label_tensor: label的tensor
    :param iterator_num: 最大迭代次数
    :param reader: 读取数据的obj
    :param summary_path: 存储log文件的位置
    :param restore: 是否从某一个断点开始训练，如果不为None,则restore[path]代表的就是之前保存模型的路径
    :return:None
    '''

    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
    center_feature, adversarial_feature = inference(images_tensor, is_training_tensor, need_featuremap=True)
    dim = center_feature.get_shape().as_list()[-1]
    print center_feature, center_feature.get_shape().as_list()
    print dim
    centers_value = np.zeros([category_num, dim], dtype=np.float32)
    centers_tensor = tf.placeholder(dtype=tf.float32, shape=np.shape(centers_value))
    global_step_tensor = tf.Variable(initial_value=0, trainable=False)
    mean_loss = tf.losses.mean_squared_error(adversarial_feature, tf.reshape(images_tensor, [-1, 784]))
    distance_category, center_loss = calculate_centerloss(center_feature, label_tensor,
                                                          centers_tensor=centers_tensor)
    loss_tensor = tf.add(tf.multiply(center_loss, _lambda), mean_loss)
    tf.summary.scalar('mean loss', mean_loss)
    tf.summary.scalar('center loss', center_loss)
    tf.summary.scalar('loss', loss_tensor)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_tensor, global_step=global_step_tensor)
    owner_step = tf.py_func(update_centers, [centers_tensor, center_feature, label_tensor, category_num], tf.float32)
    print 'owner_step is', owner_step
    with tf.control_dependencies([train_step, owner_step]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if restore is not None:
            full_path = tf.train.latest_checkpoint(restore['path'])
            print 'load model from ', full_path
            saver.restore(sess, full_path)
        train_summary_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), graph=sess.graph)
        val_summary_writer = tf.summary.FileWriter(os.path.join(summary_path, 'val'), graph=sess.graph)
        train_generator = reader.train_generator
        val_generator = reader.val_generator
        merged_summary_op = tf.summary.merge_all()
        for i in range(iterator_num):
            step_value = sess.run(global_step_tensor)

            train_images_batch, train_labels_batch = train_generator.next()
            # 反向传播的同时更新center value
            _, loss_value, merged_summary_value, centers_value, center_loss_value, mean_loss_value = sess.run(
                [train_op, loss_tensor, merged_summary_op, owner_step, center_loss,
                 mean_loss], feed_dict={
                    images_tensor: train_images_batch,
                    label_tensor: train_labels_batch,
                    is_training_tensor: True,
                    centers_tensor: centers_value
                })
            train_summary_writer.add_summary(merged_summary_value, global_step=step_value)
            if step_value % 1000 == 0:
                val_images_batch, val_labels_batch = val_generator.next()
                loss_value, merged_summary_value, centers_value = sess.run(
                    [loss_tensor, merged_summary_op, owner_step],
                    feed_dict={
                        images_tensor: val_images_batch,
                        label_tensor: val_labels_batch,
                        is_training_tensor: False,
                        centers_tensor: centers_value
                    })
                val_summary_writer.add_summary(merged_summary_value, step_value)
                print 'step: %d, validation loss: %.2f' % (
                step_value, loss_value)
                saver.save(sess, './parameters/parameters.ckpt', global_step=global_step_tensor)
            if step_value % 100 == 0:
                print 'step: %d, training loss: %.2f, center_loss_value: %.2f, mean loss value: %.2f' % (
                step_value, loss_value, center_loss_value, mean_loss_value)
                # print centers_value
        train_summary_writer.close()
        val_summary_writer.close()


def calculate_centerloss(x_tensor, label_tensor, centers_tensor):
    '''
    计算center loss
    :param x_tensor: batch的features
    :param label_tensor: label
    :param centers_tensor: centers
    :return:
    '''
    centers_batch_tensor = tf.gather(centers_tensor, tf.argmax(label_tensor, axis=1))
    loss = tf.nn.l2_loss(x_tensor - centers_batch_tensor)
    return loss, loss


if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 28, 28, 1], name='input_x')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 10], name='input_y')
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters'
    train(image_tensor, label_tensor, int(1e6), reader, restore=None)

