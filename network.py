# -*- coding: UTF-8 -*-
import tensorflow as tf
import math


def inference_ds(input_u, input_v, prop, phase, EPIWidth, disp_precision):
    output_size = int(4 / disp_precision) + 1
    u_net = inference(input_u, prop, phase, EPIWidth, disp_precision, 'u-net')
    v_net = inference(input_v, prop, phase, EPIWidth, disp_precision, 'v-net')
    concat = tf.concat([u_net, v_net], 1)
    output = fc(concat, 1024, output_size, 'FullyConnection_2')
    output = tf.nn.softmax(output)

    return output


def inference(image_pl, prop, phase, EPIWidth, disp_precision, net_name):
    with tf.name_scope(net_name):
        hidden1 = conv2d(image_pl, [2, 2, 1, 32], 'Convolution_1', phase, BN=True)
        hidden1_1 = conv2d(hidden1, [2, 2, 32, 64], 'Convolution_1_1', phase, BN=True)
        hidden1_2 = conv2d(hidden1_1, [2, 2, 64, 128], 'Convolution_1_2', phase, BN=True)
#        pool1 = pool(hidden1_2, [1, 1, 2, 1], [1, 1, 2, 1], 'Max_Pooling_1')

        hidden2 = conv2d(hidden1_2, [2, 2, 128, 256], 'Convolution_2', phase, BN=True)
        hidden2_1 = conv2d(hidden2, [2, 2, 256, 512], 'Convolution_2_1', phase, BN=True)
        hidden2_2 = conv2d(hidden2_1, [2, 2, 512, 1024], 'Convolution_2_2', phase, BN=True)
#        pool2 = pool(hidden2_2, [1, 1, 2, 1], [1, 1, 2, 1], 'Max_Pooling_2')
        pool2 = hidden2_2
        pool2_shape = pool2.get_shape()
        fc1_input_size = int(pool2_shape[1] * pool2_shape[2] * pool2_shape[3])
        pool2_resize = tf.reshape(pool2, [-1, fc1_input_size])

        hidden3 = fc(pool2_resize, fc1_input_size, 512, 'FullyConnection_1', wd=0.004)
        hidden3_drop = tf.nn.dropout(hidden3, prop)

    return hidden3_drop


def loss(logits, labels):
    #    labels = tf.to_int64(labels)
#    Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    one_hot_label = tf.one_hot(labels, 58)
#    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label, logits=logits))
    Loss = -tf.reduce_mean(one_hot_label*tf.log(logits))

    return Loss


def training(loss, learning_rate, global_step):
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    tf.summary.scalar('loss', loss)

    lr = tf.train.exponential_decay(learning_rate, global_step, 12500, 0.1, staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)
    #    optimizer = tf.train.MomentumOptimizer(lr,0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits):
    return tf.nn.top_k(logits)


'''------------------------------以下为辅助函数-------------------------------------'''


def conv2d(input_tensor, kernel_size, layer_name, phase, BN=False, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(kernel_size, stddev=1e-2))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(1e-2, shape=[kernel_size[3]]))
        with tf.name_scope('preactivate'):
            preactivate = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='VALID') + biases
        if BN:
            preactivate = batch_norm(preactivate, kernel_size[3], phase)
        activations = act(preactivate, name='activation')
        return activations


def batch_norm(input, output_size, phase_train):
    with tf.variable_scope('batch_normalization'):
        beta = tf.Variable(tf.constant(0.0, shape=[output_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[output_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
    return normed


def pool(input_tensor, kernel_size, strides, layer_name):
    with tf.name_scope(layer_name):
        output = tf.nn.max_pool(input_tensor, ksize=kernel_size, strides=strides, padding='VALID')
        return output


def fc(input_tensor, input_size, output_size, layer_name, act=tf.nn.relu, wd=0.0):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(
                tf.truncated_normal([input_size, output_size], stddev=1.0 / math.sqrt(float(input_size))))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(1e-2, shape=[output_size]))
        with tf.name_scope('preactivate'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return activations
