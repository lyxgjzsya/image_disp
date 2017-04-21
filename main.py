# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset
import numpy as np


EPIWidth = 33
batch_size = 128
test_batch = 2048
main_path = '/home/luoyaox/Work'
#main_path = '/home/cs505/workspace/luo_space'
summary_path = main_path+'/image_disp/summary'
checkpoint_path = main_path+'/image_disp/checkpoint'
disp_precision = 0.07


def trans(x):
    r = int((x+2)/disp_precision)
    return r

def do_eval_true(sess, eval, images_pl, prop, data_set,v,pro_v):
    while data_set.index_of_image < 4:#data_set.num_of_path:
        true_count = 0
        steps_per_epoch = data_set.num_examples // test_batch
        num_example = steps_per_epoch * test_batch
        for step in xrange(steps_per_epoch):
            labels_pl = tf.placeholder(tf.float32, shape=None)
            feed_dict = fill_feed_dict(data_set,images_pl,labels_pl,prop,'test',v,pro_v)
            output, label = sess.run([eval, labels_pl],feed_dict=feed_dict)
            for i in xrange(test_batch):
                disp = (output[1][i]*disp_precision)-2+disp_precision/2
                true_count += abs(disp-label[i])<0.07
        precision = float(true_count) / num_example
        print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))
    data_set.set_index_of_image(0)


def fill_feed_dict(data_sets, images_placeholder, labels_placeholder, prop_placeholder, mode,im_v,pro_v):
    count = batch_size
    if mode == 'test':
        count = test_batch
    images, v, labels = data_sets.next_batch(count)
    prop = 0.5
    if mode == 'test':
        prop = 1
    elif mode == 'train':
        #训练时label转为class
        labels = map(trans,labels)
    feed_dict = {
        images_placeholder: images,
        labels_placeholder: labels,
        prop_placeholder: prop,
        pro_v:prop,
        im_v:v,
    }
    return feed_dict


def main():
    with tf.Graph().as_default():
        train_sets = dataset.get_datasets(main_path, EPIWidth, disp_precision, 'train')
        test_sets = dataset.get_datasets(main_path, EPIWidth, disp_precision, 'test')

        global_step = tf.Variable(0, trainable=False)

        images_placeholder_v = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 3))
        images_placeholder_u = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=None)
        prop_placeholder_v = tf.placeholder('float')
        prop_placeholder_u = tf.placeholder('float')

#        logits = network.inference(images_placeholder, prop_placeholder, EPIWidth, disp_precision,'my_net')
        logits = network.inference_v2(images_placeholder_u,images_placeholder_v,prop_placeholder_u,prop_placeholder_v,EPIWidth,disp_precision)

        loss = network.loss(logits, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        eval = network.evaluation(logits)

        summary = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
#            saver.restore(sess,checkpoint_path+'/model.ckpt')#从其他平台训练的结果
#            saver.restore(sess,ckpt.model_checkpoint_path)#本地训练的结果
            print ("restore from checkpoint!")
        else:
            print("no checkpoint found!")

        start_time = time.time()

        for step in xrange(100000):

            feed_dict = fill_feed_dict(train_sets, images_placeholder_u, labels_placeholder, prop_placeholder_u,'train',images_placeholder_v,prop_placeholder_v)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 50000 == 49999:
                saver.save(sess, checkpoint_path+'/model.ckpt',global_step=step)
                print('Training Data Eval:')
                do_eval_true(sess,eval,images_placeholder_u,prop_placeholder_u,test_sets,images_placeholder_v,prop_placeholder_v)



if __name__ == '__main__':
    main()

    #通道叠加测试
    a=np.arange(0,72)
    a=a.reshape([2,2,3,3,2])
    b=np.arange(0,72)
    b=b.reshape([2,2,3,3,2])

    a=a.reshape([36,2])
    b=b.reshape([36,2])
    result1 = np.column_stack((a,b))
    result1 = result1.reshape([2,2,3,3,4])


    result1 = result1.reshape([36,4])
    a1,b1 = np.hsplit(result1,2)
    a1=a1.reshape([2,2,3,3,2])
    b1=b1.reshape([2,2,3,3,2])



    print 'done'



