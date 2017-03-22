# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset


def fill_feed_dict(data_sets, images_placeholder, labels_placeholder, fake=False):
    images = None
    labels = None
    if fake:
        images_tmp = [1] * 9 * 16 * 3
        images = [images_tmp]
        labels = [1]
    else:
        images, labels = data_sets.next_batch(1)#batch 目前设置大于1了就有bug，还没修复
    feed_dict = {
        images_placeholder: images,
        labels_placeholder: labels,
    }
    return feed_dict,labels


def main():
    data_sets = dataset.get_datasets('/home/luoyaox/Work/box')
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_placeholder = tf.placeholder(tf.float32, shape=(None, 9 * 16 * 3))
        labels_placeholder = tf.placeholder(tf.float32, shape=(None))

        logits = network.inference(images_placeholder)

        loss = network.loss(logits, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        # eval_correct = network.evaluation(logits,labels_placeholder) 判断准确率还没做

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        for step in xrange(100000):
            start_time = time.time()
            feed_dict,label = fill_feed_dict(data_sets, images_placeholder, labels_placeholder)
            _, loss_value, output = sess.run([train_op, loss, logits], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                print ('Output:%f  Label:%f' % (output[0],label[0]))#没做准确率计算，姑且先显示当前这次的网络输出和label



if __name__ == '__main__':
    main()

