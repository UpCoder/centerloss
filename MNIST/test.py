import tensorflow as tf
from model import inference
import numpy as np
from reader import Reader
from plot import plot_scatter


def test(images_tensor, labels_tensor, reader, parameters_path):
    generator = reader.test_generator
    logits, featuremap_tensor = inference(images_tensor, False, True)
    prediction_tensor = tf.argmax(logits, 1)
    accuracy_tensor = tf.reduce_mean(tf.cast(tf.equal(prediction_tensor, tf.argmax(labels_tensor, 1)), tf.float32))
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
            prediction, accuracy, featuremap_value = sess.run([prediction_tensor, accuracy_tensor, featuremap_tensor], feed_dict={
                images_tensor: images_batch,
                labels_tensor: labels_batch
            })
            featuremaps.extend(featuremap_value)
            print accuracy
    print np.shape(featuremaps), np.shape(labels)
    featuremaps = np.array(featuremaps)
    # featuremaps = np.array(featuremaps) / 1000.0
    # print featuremaps
    plot_scatter(featuremaps[:, 0], featuremaps[:, 1], labels, 10)

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    images_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_x')
    labels_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    test(images_tensor, labels_tensor, reader, '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters')