# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset
import numpy as np
EPIWidth = 32

def fill_feed_dict(data_sets, images_placeholder, labels_placeholder, fake=False):
    images = None
    labels = None
    if fake:
        images_tmp = [1] * 9 * EPIWidth * 3
        images = [images_tmp]
        labels = [1]
    else:
        images, labels = data_sets.next_batch(50)
    feed_dict = {
        images_placeholder: images,
        labels_placeholder: labels,
    }
    return feed_dict,labels


def main():
    data_sets = dataset.get_datasets('/home/luoyaox/Work/box')
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_placeholder = tf.placeholder(tf.float32, shape=(None, 9 * EPIWidth * 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(None))

        logits = network.inference_old(images_placeholder)

        loss = network.loss(logits, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        eval_correct = network.evaluation(logits,labels_placeholder)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        start_time = time.time()

        for step in xrange(100001):

            feed_dict,labels = fill_feed_dict(data_sets, images_placeholder, labels_placeholder)
            _, loss_value,output = sess.run([train_op, loss, logits], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
#                print ('Output:%f  Label:%f' % (output[0],label[0]))#没做准确率计算，姑且先显示当前这次的网络输出和label
            #    print output[0:2]
            #    print feed_dict[labels_placeholder][0:2]
            if step % 1000 == 0:
                feed_dict = {
                    images_placeholder: data_sets.images,
                    labels_placeholder: data_sets.labels,
                }
                _,output = sess.run([loss,logits],feed_dict=feed_dict)
                true_count=0
                for i in xrange(262144):
                    max = -100
                    no = 0
                    for j in xrange(41):
                        if output[i][j]>max:
                            max=output[i][j]
                            no=j
                    if no==data_sets.labels[i]:
                        true_count+=1
                print float(true_count)/262144.0









if __name__ == '__main__':
    main()



