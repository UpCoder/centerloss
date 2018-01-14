from reader import Reader
import tensorflow as tf
from train import train

if __name__ == '__main__':
    reader = Reader('/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/data')
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_x')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters'
    train(image_tensor, label_tensor, int(1e6), reader, restore=None)