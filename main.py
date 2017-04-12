# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset
import os.path
import numpy as np
EPIWidth = 33
batch_size = 200
box_path = '/home/luoyaox/Work/box'
disp_precision = 0.07

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,prop_placeholder,data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_example = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder,prop_placeholder,mode='test')
        true_count += sess.run(eval_correct,feed_dict=feed_dict)
    precision = float(true_count)/num_example
    print ('example: %d, correct: %d, Precision: %0.04f' % (num_example,true_count,precision))


def fill_feed_dict(data_sets, images_placeholder, labels_placeholder, prop_placeholder, mode='train'):
    images, labels = data_sets.next_batch(batch_size)
    prop = 0.5
    if mode=='test':
        prop = 1
    feed_dict = {
        images_placeholder: images,
        labels_placeholder: labels,
        prop_placeholder: prop,
    }
    return feed_dict


def main():
    data_sets = dataset.get_datasets(box_path,EPIWidth,disp_precision)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_placeholder = tf.placeholder(tf.float32, shape=(None, 9 , EPIWidth , 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        prop_placeholder = tf.placeholder('float')

        logits = network.inference_old(images_placeholder,prop_placeholder,EPIWidth,disp_precision)

        loss = network.loss(logits, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        eval_correct = network.evaluation(logits,labels_placeholder)

        saver = tf.train.Saver()

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        for step in xrange(100000):

            feed_dict = fill_feed_dict(data_sets, images_placeholder, labels_placeholder, prop_placeholder)
            _, loss_value,output = sess.run([train_op, loss, logits], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))


            if step % 10000 == 0 :
                if step != 0:
#                    saver.save(sess, "/home/luoyaox/Work/box/model.ckpt")
                    print('Training Data Eval:')
                    do_eval(sess,eval_correct,images_placeholder,labels_placeholder,prop_placeholder,data_sets)



if __name__ == '__main__':
    main()




