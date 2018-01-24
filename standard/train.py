import tensorflow as tf
from model import inference, inference_LeNet
from reader import Reader
import numpy as np
import os


def owner_function():
    print 'hello'


def train(images_tensor, label_tensor, iterator_num, reader, summary_path='./log', restore=None):
    is_training_tensor = tf.placeholder(dtype=tf.bool, shape=[], name='training_flag')
    logits = inference(images_tensor, is_training_tensor)
    global_step_tensor = tf.Variable(initial_value=0, trainable=False)
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits))
    tf.summary.scalar('loss', softmax_loss)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(softmax_loss, global_step=global_step_tensor)
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op('train')

    predicted_tensor = tf.argmax(logits, 1)
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
            if step_value % 1000 == 0:
                val_images_batch, val_labels_batch = val_generator.next()
                validation_acc, loss_value, merged_summary_value = sess.run(
                    [accuracy_tensor, softmax_loss, merged_summary_op],
                    feed_dict={
                        images_tensor: val_images_batch,
                        label_tensor: val_labels_batch,
                        is_training_tensor: False
                    })
                val_summary_writer.add_summary(merged_summary_value, step_value)
                print 'step: %d, validation accuracy: %.2f, validation loss: %.2f' % (
                step_value, validation_acc, loss_value)
                saver.save(sess, './parameters/parameters.ckpt', global_step=global_step_tensor)
            train_images_batch, train_labels_batch = train_generator.next()
            _, train_acc, train_prediction, loss_value, merged_summary_value = sess.run(
                [train_op, accuracy_tensor, predicted_tensor, softmax_loss, merged_summary_op], feed_dict={
                    images_tensor: train_images_batch,
                    label_tensor: train_labels_batch,
                    is_training_tensor: True
                })
            train_summary_writer.add_summary(merged_summary_value, global_step=step_value)
            if step_value % 100 == 0:
                print 'step: %d, training accuracy: %.2f, training loss: %.2f' % (step_value, train_acc, loss_value)
        train_summary_writer.close()
        val_summary_writer.close()

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_x')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters'
    train(image_tensor, label_tensor, int(1e6), reader, restore=None)

