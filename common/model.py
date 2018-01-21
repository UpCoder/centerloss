# -*- coding=utf-8 -*-
import tensorflow as tf


def parametric_relu(_x):
    '''
    implement pReLE
    :param _x:
    :return:
    '''
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def do_conv(x, layer_name, kernel_size, depth, stride_size, padding='SAME', is_activation=True, activation_method=None, is_bn=True, config=None):
    shape = x.get_shape().as_list()
    if len(shape) == 3:
        shape.extend(1)
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[kernel_size, kernel_size, shape[-1], depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', shape=[depth], initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.conv2d(x, weight, strides=[1, stride_size, stride_size, 1], padding=padding)

        if is_bn:
            output = batch_norm_layer(output, config['is_training'], scope='batch_norm_layer')
        else:
            output = tf.nn.bias_add(output, bias)
        if is_activation:
            output = activation_method(output)
    return output


def do_pooling(x, method, layer_name, kernel_size, stride_size, padding='SAME'):
    with tf.variable_scope(layer_name):
        if method == 'max':
            output = tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                                    padding=padding)
        else:
            output = tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                                    padding=padding)
    return output


def do_fc(x, layer_name, depth, is_activation=True, activation_method=None):
    shape = x.get_shape().as_list()
    if len(shape) == 3:
        shape.extend(1)
    with tf.variable_scope(layer_name):
        weight = tf.get_variable('weight', shape=[shape[-1], depth],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', shape=[depth], initializer=tf.constant_initializer(value=0.0))
        output = tf.matmul(x, weight)
        output = tf.nn.bias_add(output, bias)
        if is_activation:
            output = activation_method(output)
    return output


def flatten22Dim(input):
    '''
    输入是一个４维的tensor
    :param input:
    :return:
    '''
    shape = input.get_shape().as_list()
    print shape
    nodes = shape[1] * shape[2] * shape[3]
    print nodes
    if shape[0] is None:
        result = tf.reshape(input, [-1, nodes])
    else:
        result = tf.reshape(input, [shape[0], nodes])
    return result


def batch_norm_layer(inputs, phase_train, scope=None):
    return tf.cond(phase_train,
                   lambda: tf.contrib.layers.batch_norm(inputs, is_training=True, scale=True,
                                                        updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(inputs, is_training=False, scale=True,
                                                        updates_collections=None, scope=scope, reuse=True))


def inference(image_tensor, is_training, need_featuremap=False, output_num=10):
    config = {}
    config['is_training'] = tf.convert_to_tensor(is_training,
                                                 dtype='bool',
                                                 name='is_training')
    activation_method = parametric_relu
    layer1 = do_conv(image_tensor, 'layer1-conv1', kernel_size=5, depth=32, stride_size=1, padding='SAME',
                     config=config, activation_method=activation_method)
    layer2 = do_conv(layer1, 'layer2-conv2', kernel_size=5, depth=32, stride_size=1, padding='SAME', config=config,
                     activation_method=activation_method)
    layer3 = do_pooling(layer2, 'max', 'layer3-maxpooling1', kernel_size=2, stride_size=2, padding='VALID')

    layer4 = do_conv(layer3, 'layer4-conv3', kernel_size=5, depth=64, stride_size=1, padding='SAME', config=config,
                     activation_method=activation_method)
    layer5 = do_conv(layer4, 'layer5-conv4', kernel_size=5, depth=64, stride_size=1, padding='SAME', config=config,
                     activation_method=activation_method)
    layer6 = do_pooling(layer5, 'max', 'layer6-maxpooling2', kernel_size=2, stride_size=2, padding='VALID')

    layer7 = do_conv(layer6, 'layer7-conv5', kernel_size=5, depth=128, stride_size=1, padding='SAME', config=config,
                     activation_method=activation_method)
    layer8 = do_conv(layer7, 'layer8-conv6', kernel_size=5, depth=128, stride_size=1, padding='SAME', config=config,
                     activation_method=activation_method)
    layer9 = do_pooling(layer8, 'max', 'layer9-maxpooling3', kernel_size=2, stride_size=2, padding='VALID')

    print layer9
    conv_output = flatten22Dim(layer9)

    fc1 = do_fc(conv_output, layer_name='layer13-fc1', depth=2, activation_method=activation_method)
    # if train:
    #     layer10 = tf.nn.dropout(layer10, 0.5)
    fc2 = do_fc(fc1, layer_name='layer14-fc2', depth=output_num, is_activation=False, activation_method=activation_method)
    if need_featuremap:
        return fc2, fc1
    return fc2


def inference_LeNet(image_tensor, train):
    layer1 = do_conv(image_tensor, 'layer1-conv1', kernel_size=5, depth=5, stride_size=1, padding='SAME')

    layer2 = do_pooling(layer1, method='max', layer_name='layer2-maxpooling1', kernel_size=2, stride_size=2, padding='SAME')

    layer3 = do_conv(layer2, layer_name='layer3-conv2', kernel_size=5, depth=64, stride_size=1, padding='SAME')

    layer4 = do_pooling(layer3, method='max', layer_name='layer4-maxpooling2', kernel_size=2, stride_size=2, padding='SAME')

    layer4 = flatten22Dim(layer4)

    layer5 = do_fc(layer4, layer_name='layer5-fc1', depth=512)
    if train:
        layer5 = tf.nn.dropout(layer5, 0.5)
    layer6 = do_fc(layer5, layer_name='layer6-fc2', depth=10)

    return layer6
if __name__ == '__main__':
    image_tensor = tf.placeholder(dtype=tf.float32, shape=[100, 28, 28, 1])
    inference(image_tensor)