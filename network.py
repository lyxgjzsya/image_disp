# -*- coding: UTF-8 -*-
import tensorflow as tf

#测试用小型网络
def inference(image_pl):
    image_placeholder = tf.reshape(image_pl,[-1,9,16,3])
    w_conv1 = tf.Variable(tf.truncated_normal([1, 3, 3, 32], stddev=1e-2))
    conv1 = tf.nn.conv2d(image_placeholder, w_conv1, [1, 1, 1, 1], padding='VALID')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    a_conv1 = tf.nn.relu(conv1 + b_conv1)
    pool1 = tf.nn.max_pool(a_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    w_conv2 = tf.Variable(tf.truncated_normal([1, 3, 32, 32], stddev=1e-2))
    conv2 = tf.nn.conv2d(pool1, w_conv2, [1, 1, 1, 1], padding='VALID')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
    a_conv2 = tf.nn.relu(conv2 + b_conv2)
    pool2 = tf.nn.max_pool(a_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

    w_fc1 = tf.Variable(tf.truncated_normal([9 * 3 * 32, 100], stddev=1e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[100]))
    pool2_tmp = tf.reshape(pool2, [-1, 9 * 3 * 32])
    a_fc1 = tf.nn.relu(tf.matmul(pool2_tmp, w_fc1) + b_fc1)

    w_fc2 = tf.Variable(tf.truncated_normal([100, 1], stddev=1e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
    a_fc2 = tf.tanh(tf.matmul(a_fc1, w_fc2) + b_fc2)*4

    print 'image:',image_placeholder.get_shape()
    print 'a_conv1:',a_conv1.get_shape()
    print 'pool1',pool1.get_shape()
    print 'a_conv2',a_conv2.get_shape()
    print 'pool2',pool2.get_shape()
    print 'a_fc1',a_fc1.get_shape()
    print 'a_fc2',a_fc2.get_shape()

    return a_fc2


def loss(logits, labels_placeholder):
    Loss = tf.reduce_sum(tf.pow(logits - labels_placeholder, 2))

    return Loss


def training(loss, learning_rate, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


