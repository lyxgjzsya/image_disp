# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset
import numpy as np
import scipy.io as sio


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

def do_eval_true(sess, eval, images_u, image_v, prop, phase_train, data_set):
    count = 0
    while count < data_set.num_of_path:
        true_count = 0
        steps_per_epoch = data_set.num_examples // test_batch
        num_example = steps_per_epoch * test_batch
        for step in xrange(steps_per_epoch):
            labels_pl = tf.placeholder(tf.float32, shape=None)
            feed_dict = fill_feed_dict(data_set,images_u,image_v,labels_pl,prop,phase_train,'test')
            output, label = sess.run([eval, labels_pl],feed_dict=feed_dict)
            for i in xrange(test_batch):
                disp = (output[1][i]*disp_precision)-2+disp_precision/2
                true_count += abs(disp-label[i])<0.07
        precision = float(true_count) / num_example
        print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))
        count += 1


def fill_feed_dict(data_sets, images_u_pl, image_v_pl, labels_placeholder, prop_placeholder,phase_train, mode='train'):
    count = batch_size
    if mode == 'test':
        count = test_batch
    images_u, images_v, labels = data_sets.next_batch(count)
    if mode == 'test':
        feed_dict = {
            images_u_pl: images_u,
            image_v_pl: images_v,
            labels_placeholder: labels,
            prop_placeholder:1.0,
            phase_train:False,
        }
    elif mode == 'train':
        feed_dict = {
            images_u_pl: images_u,
            image_v_pl: images_v,
            labels_placeholder: map(trans,labels),
            prop_placeholder:0.5,
            phase_train:True,
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
        prop_placeholder = tf.placeholder('float')
        phase_train = tf.placeholder(tf.bool,name='phase_train')

        logits = network.inference_ds(images_placeholder_u,images_placeholder_v,prop_placeholder,phase_train,EPIWidth,disp_precision)

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
#            saver.restore(sess,checkpoint_path+'/model.ckpt')#利用不同平台的训练结果
#            saver.restore(sess,ckpt.model_checkpoint_path)#本地训练的结果
            print ("restore from checkpoint!")
        else:
            print("no checkpoint found!")

        start_time = time.time()

        for step in xrange(50000):

            feed_dict = fill_feed_dict(train_sets, images_placeholder_u, images_placeholder_v, labels_placeholder, prop_placeholder,phase_train,'train')
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 25000 == 24999:
                saver.save(sess, checkpoint_path+'/model.ckpt',global_step=step)
                print('Training Data Eval:')
                do_eval_true(sess,eval,images_placeholder_u,images_placeholder_v,prop_placeholder,phase_train,test_sets)



if __name__ == '__main__':
#    main()
    a = np.arange(12)
    a = a.reshape([2,2,3])
    sio.savemat('test.mat',{'matrix':a})

    print 'done'


