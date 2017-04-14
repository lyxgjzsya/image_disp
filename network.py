# -*- coding: UTF-8 -*-
import tensorflow as tf
import math


# 测试用小型网络
def inference(image_pl,prop, EPIWidth, disp_precision):
    output_size = int(4 / disp_precision) + 1

    hidden1 = conv2d(image_pl,[3,3,3,64],'Convolution_1')
    pool1 = pool(hidden1,[1,1,2,1],[1,1,2,1],'Max_Pooling_1')

    hidden2 = conv2d(pool1,[3,3,64,128],'Convolution_2')
    pool2 = pool(hidden2,[1,1,2,1],[1,1,2,1],'Max_Pooling_2')

    pool2_shape = pool2.get_shape()
    fc1_input_size = int(pool2_shape[1] * pool2_shape[2] * pool2_shape[3])
    pool2_resize = tf.reshape(pool2, [-1, fc1_input_size])

    hidden3 = fc(pool2_resize,fc1_input_size,1024,'FullyConnection_1')
    hidden3_drop = tf.nn.dropout(hidden3,prop)

    output = fc(hidden3_drop,1024,output_size,'FullyConnection_2')

    print 'image:', image_pl.get_shape()
    print 'a_conv1:', hidden1.get_shape()
    print 'pool2', pool1.get_shape()
    print 'a_conv2', hidden2.get_shape()
    print 'a_fc1', pool2.get_shape()
    print 'a_fc2', hidden3.get_shape()

    return output


def loss(logits, labels):
#    labels = tf.to_int64(labels)
    Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    return Loss


def training(loss, learning_rate, global_step):
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tf.summary.scalar('loss',loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluationv2(logits):
    return tf.nn.top_k(logits)

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)

    return tf.reduce_sum(tf.cast(correct, tf.int32))


def conv2d(input_tensor, kernel_size, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(kernel_size, stddev=1e-2))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(1e-2,shape=[kernel_size[3]]))
        with tf.name_scope('preactivate'):
            preactivate = tf.nn.conv2d(input_tensor,weights,[1, 1, 1, 1],padding='VALID')+biases
        activations = act(preactivate, name='activation')
        return activations


def pool(input_tensor, kernel_size, strides, layer_name):
    with tf.name_scope(layer_name):
        output = tf.nn.max_pool(input_tensor, ksize=kernel_size, strides=strides, padding='VALID')
        return output


def fc(input_tensor, input_size, output_size, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=1.0 / math.sqrt(float(input_size))))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(1e-2, shape=[output_size]))
        with tf.name_scope('preactivate'):
            preactivate = tf.matmul(input_tensor,weights)+biases
        activations = act(preactivate, name='activation')
        return activations


def inference_test(image_pl,prop, EPIWidth, disp_precision):
    '''
     local test
    '''
    output_size = int(4 / disp_precision) + 1
    tf.summary.image('input',image_pl)
    input=tf.reshape(image_pl,[-1,9*33*3])

    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([891, 256],
                                stddev=1.0 / math.sqrt(float(891))),
            name='weights')
        biases = tf.Variable(tf.zeros([256]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([256, 1024],
                                stddev=1.0 / math.sqrt(float(256))),
            name='weights')
        biases = tf.Variable(tf.zeros([1024]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([1024, output_size],
                                stddev=1.0 / math.sqrt(float(1024))),
            name='weights')
        biases = tf.Variable(tf.zeros([output_size]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits