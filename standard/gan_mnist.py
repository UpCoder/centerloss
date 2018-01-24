# -*－coding=utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

print 'ok'
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

h2_size = 300   # 第二个隐藏层的节点个数
h1_size = 150   # 第一个隐藏层的节点个数

#判别器模型
X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, h2_size]))
D_b1 = tf.Variable(tf.zeros(shape=[h2_size]))

D_W2 = tf.Variable(xavier_init([h2_size, h1_size]))
D_b2 = tf.Variable(tf.zeros(shape=[h1_size]))

D_W3 = tf.Variable(xavier_init([h1_size, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


#生成器模型
Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, h1_size]))
G_b1 = tf.Variable(tf.zeros(shape=[h1_size]))

G_W2 = tf.Variable(xavier_init([h1_size, h2_size]))
G_b2 = tf.Variable(tf.zeros(shape=[h2_size]))

G_W3 = tf.Variable(xavier_init([h2_size, 784]))
G_b3 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


#z是100维随机向量
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)   # G_h1是128维
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3    #  G_log_prob是784维 ，也就是生成的图片
    G_prob = tf.nn.sigmoid(G_log_prob)    #  用sigmoid函数激活

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)   #得到的是128维
    D_h1 = tf.nn.dropout(D_h1, 0.7)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h2 = tf.nn.dropout(D_h2, 0.7)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3    #得到的是1维输出
    D_prob = tf.nn.sigmoid(D_logit)    #使用sigmoid函数激活

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(10,10))    #4*4=16个图
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')

    return fig   #根据samples画图


G_sample = generator(Z)     #通过z生成 784维向量
D_real, D_logit_real = discriminator(X)   #分别得到真实样本的输出概率，1为输出
D_fake, D_logit_fake = discriminator(G_sample)   #得到假的哦


#论文中的公式  GAN的理论公式  也就是优化目标
D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real, 1e-10, 1.0)) + tf.log(tf.clip_by_value((1. - D_fake), 1e-10, 1.0)))
G_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_fake, 1e-10, 1.0)))

# Alternative losses:
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))



#按照公式优化
D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)   #读取mnist数据

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        # print it
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})    # 16*100维的向量送到Z，得到G_sample
        #最后得到16*784维的存到samples

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
print()