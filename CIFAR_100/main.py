from common.train import train
from Reader import Reader
import tensorflow as tf

if __name__ == '__main__':
    reader = Reader(data_dir='/home/give/Documents/dataset/cifar/cifar-100-python', batch_size=100,
                    reshape_flag=True)
    category_num = 100
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 32, 32, 3], name='input_x')
    label_tensor = tf.placeholder(dtype=tf.float32, shape=[100, category_num], name='input_y')
    restore_obj = dict()
    restore_obj['path'] = '/home/give/PycharmProjects/Reproduce/CenterLoss/MNIST/parameters'
    train(image_tensor, label_tensor, int(1e6), reader, restore=None, output_num=category_num)
