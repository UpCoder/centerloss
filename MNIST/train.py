# -*- coding=utf-8 -*-
import tensorflow as tf
from model import inference, inference_LeNet
from reader import Reader
import numpy as np
import os
category_num = 10
_lambda = 1
_alpha = 0.5


def update_centers(centers, data, labels, category_num):
    '''
    更新质心
    :param centers: 所有质心的数组 type: numpy.array shape: [category_num, 2]
    :param data: 一个batch数据的features
    :param labels: 一个batch数据的label one hot编码格式
    :param category_num: 一个有多少种label，在这里就是10
    :return: 更新之后的centers
    '''
    centers = np.array(centers)
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
    centers_value = np.zeros([category_num, 2], dtype=np.float32)
    centers_tensor = tf.placeholder(dtype=tf.float32, shape=np.shape(centers_value))
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
    logits, featuresmap = inference(images_tensor, is_training_tensor, need_featuremap=True)
    predicted_tensor = tf.argmax(logits, 1)
    global_step_tensor = tf.Variable(initial_value=0, trainable=False)
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits))
    distance_category, center_loss = calculate_centerloss(featuresmap, label_tensor,
                                                          centers_tensor=centers_tensor)
    loss_tensor = tf.add(tf.multiply(center_loss, _lambda), softmax_loss)
    tf.summary.scalar('softmax loss', softmax_loss)
    tf.summary.scalar('center loss', center_loss)
    tf.summary.scalar('loss', loss_tensor)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss_tensor, global_step=global_step_tensor)
    owner_step = tf.py_func(update_centers, [centers_tensor, featuresmap, label_tensor, category_num], tf.float32)
    print 'owner_step is', owner_step
    with tf.control_dependencies([train_step, owner_step]):
        train_op = tf.no_op('train')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_tensor, 1))
    accuracy_tensor = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_tensor)

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
            _, train_acc, train_prediction, loss_value, merged_summary_value, centers_value, center_loss_value, softmax_loss_value, distance_category_value = sess.run(
                [train_op, accuracy_tensor, predicted_tensor, loss_tensor, merged_summary_op, owner_step, center_loss,
                 softmax_loss, distance_category], feed_dict={
                    images_tensor: train_images_batch,
                    label_tensor: train_labels_batch,
                    is_training_tensor: True,
                    centers_tensor: centers_value
                })
            train_summary_writer.add_summary(merged_summary_value, global_step=step_value)
            if step_value % 1000 == 0:
                val_images_batch, val_labels_batch = val_generator.next()
                validation_acc, loss_value, merged_summary_value, centers_value = sess.run(
                    [accuracy_tensor, loss_tensor, merged_summary_op, owner_step],
                    feed_dict={
                        images_tensor: val_images_batch,
                        label_tensor: val_labels_batch,
                        is_training_tensor: False,
                        centers_tensor: centers_value
                    })
                val_summary_writer.add_summary(merged_summary_value, step_value)
                print 'step: %d, validation accuracy: %.2f, validation loss: %.2f' % (
                step_value, validation_acc, loss_value)
                saver.save(sess, './parameters/parameters.ckpt', global_step=global_step_tensor)
            if step_value % 100 == 0:
                print 'step: %d, training accuracy: %.2f, training loss: %.2f, center_loss_value: %.2f, softmax_loss_value: %.2f' % (
                step_value, train_acc, loss_value, center_loss_value, softmax_loss_value)
                # print centers_value
                print distance_category_value
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
    # loss_tensor = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)
    # notonehot_label_tensor = tf.argmax(label_tensor, axis=1)
    # for i in range(category_num):
    #     selected_x = tf.gather(x_tensor,
    #                            tf.cast(
    #                                tf.squeeze(
    #                                    tf.where(
    #                                        tf.equal(
    #                                            notonehot_label_tensor,
    #                                            [i]
    #                                        ),
    #                                        None,
    #                                        None
    #                                    )
    #                                ), tf.int32))
    #     selected_y = tf.convert_to_tensor(centers_tensor[i], dtype=tf.float32)
    #     selected_x2 = tf.multiply(selected_x, selected_x)
    #     selected_y2 = tf.multiply(selected_y, selected_y)
    #     selected_xy = tf.multiply(selected_x, selected_y)
    #     distance_category = tf.abs(tf.subtract(tf.add(selected_x2, selected_y2), 2*selected_xy))
    #     distance_category = tf.abs(tf.reduce_sum(0.5 * distance_category))
    #     loss_tensor = tf.abs(tf.add(loss_tensor, distance_category))
    # return loss_tensor, loss_tensor

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 28, 28, 1], name='input_x')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 10], name='input_y')
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters'
    train(image_tensor, label_tensor, int(1e6), reader, restore=None)

