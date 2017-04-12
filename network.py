# -*- coding: UTF-8 -*-
import tensorflow as tf
import math


# 测试用小型网络
def inference_old(image_pl,prop, EPIWidth, disp_precision):
    output_size = int(4 / disp_precision) + 1

#    image_placeholder = tf.reshape(image_pl, [-1, 9, EPIWidth, 3])
    w_conv1 = tf.Variable(tf.truncated_normal([2, 5, 3, 32], stddev=1e-4))
    conv1 = tf.nn.conv2d(image_pl, w_conv1, [1, 1, 1, 1], padding='VALID')
    b_conv1 = tf.Variable(tf.constant(1e-4, shape=[32]))
    a_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))
#    pool1 = tf.nn.max_pool(a_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

    w_conv2 = tf.Variable(tf.truncated_normal([1, 3, 32, 64], stddev=1e-4))
#    conv2 = tf.nn.conv2d(pool1, w_conv2, [1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.conv2d(a_conv1, w_conv2, [1, 1, 1, 1], padding='VALID')
    b_conv2 = tf.Variable(tf.constant(1e-4, shape=[64]))
    a_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
    pool2 = tf.nn.max_pool(a_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    w_conv3 = tf.Variable(tf.truncated_normal([1, 3, 64, 64], stddev=1e-4))
    conv3 = tf.nn.conv2d(a_conv2, w_conv3, [1, 1, 1, 1], padding='VALID')
    b_conv3 = tf.Variable(tf.constant(1e-4, shape=[64]))
    a_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))
    pool3 = tf.nn.max_pool(a_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    pool3_shape = pool3.get_shape()
    fc1_input_size = int(pool3_shape[1] * pool3_shape[2] * pool3_shape[3])

    w_fc1 = tf.Variable(tf.truncated_normal([fc1_input_size, 1024], stddev=1.0 / math.sqrt(float(fc1_input_size))))
    b_fc1 = tf.Variable(tf.constant(1e-4, shape=[1024]))
    pool3_tmp = tf.reshape(pool3, [-1, fc1_input_size])
    a_fc1 = tf.nn.relu(tf.matmul(pool3_tmp, w_fc1) + b_fc1)
    a_fc1_drop = tf.nn.dropout(a_fc1,prop)

    w_fc2 = tf.Variable(tf.truncated_normal([1024, output_size], stddev=1.0 / math.sqrt(float(1024))))
    b_fc2 = tf.Variable(tf.constant(0.0, shape=[output_size]))
    #    a_fc2 = tf.nn.softmax(tf.matmul(a_fc1,w_fc2) + b_fc2)
    a_fc2 = tf.matmul(a_fc1_drop, w_fc2) + b_fc2

    print 'image:', image_pl.get_shape()
    print 'a_conv1:', a_conv1.get_shape()
    print 'pool2', pool2.get_shape()
    print 'a_conv2', a_conv2.get_shape()
    print 'pool3', pool3.get_shape()
    print 'a_fc1', a_fc1.get_shape()
    print 'a_fc2', a_fc2.get_shape()

    return a_fc2


def inference(images, EPIWidth, disp_precision):
    output_size = int(4 / disp_precision) + 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([9 * EPIWidth * 3, 512],
                                stddev=1.0 / math.sqrt(float(9 * EPIWidth * 3))),
            name='weights')
        biases = tf.Variable(tf.zeros([512]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([512, 256],
                                stddev=1.0 / math.sqrt(float(512))),
            name='weights')
        biases = tf.Variable(tf.zeros([256]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([256, output_size],
                                stddev=1.0 / math.sqrt(float(256))),
            name='weights')
        biases = tf.Variable(tf.zeros([output_size]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    print 'image:', images.get_shape()
    print 'hidden1:', hidden1.get_shape()
    print 'hidden2:', hidden2.get_shape()
    print 'output:', logits.get_shape()

    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    return Loss


def training(loss, learning_rate, global_step):
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)

    return tf.reduce_sum(tf.cast(correct, tf.int32))
