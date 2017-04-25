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

    hidden3 = fc(pool2_resize,fc1_input_size,512,'FullyConnection_1',wd=0.004)
    hidden3_drop = tf.nn.dropout(hidden3,prop)

    output = fc(hidden3_drop,512,output_size,'FullyConnection_2')

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

#    num_batches_per_epoch = 200000/50
#    decay_step = int(num_batches_per_epoch*10)
    lr = tf.train.exponential_decay(learning_rate,global_step,10000,0.9,staircase=True)

    optimizer = tf.train.AdamOptimizer(lr)
#    optimizer = tf.train.MomentumOptimizer(lr,0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits):
    return tf.nn.top_k(logits)

'''------------------------------以下为辅助函数-------------------------------------'''

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


def batch_norm(input,output_size,phase_train):
    with tf.variable_scope('bn'):
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
            weights = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=1.0 / math.sqrt(float(input_size))))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(1e-2, shape=[output_size]))
        with tf.name_scope('preactivate'):
            preactivate = tf.matmul(input_tensor,weights)+biases
        activations = act(preactivate, name='activation')
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights),wd,name='weight_loss')
            tf.add_to_collection('losses',weight_decay)
        return activations


def inference_test(image_pl,prop, EPIWidth, disp_precision):
    '''
     local test
    '''
    output_size = int(4 / disp_precision) + 1

    input=tf.reshape(image_pl,[-1,9*33*6])

    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([891*2, 256],
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