# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset
import numpy as np
import scipy.io as sio
import collections
import math

EPIWidth = 9
batch_size = 128
test_batch = 2048
#main_path = '/home/luoyaox/Work/lightfield'
main_path = '/home/cs505/workspace/luo_space'
summary_path = main_path + '/image_disp/summary'
checkpoint_path = main_path + '/image_disp/checkpoint'
disp_precision = 0.07
disp_min = -4
disp_max = 4
class_num = int((disp_max - disp_min) / disp_precision) + 1

'''
attention! in this version,the label of train is class-label,while the label of test is disp-label!
'''


def do_eval_true(sess, eval, logits, images_u, images_v, prop, phase_train, data_set):
    count = 0
    print 1
    while count < data_set.num_of_path:
        true_count = 0
        start = time.time()
        output_txt = []
        raw_output_mat = []
        steps_per_epoch = data_set.num_examples // test_batch
        num_example = steps_per_epoch * test_batch
        for step in xrange(steps_per_epoch):
            labels_pl = tf.placeholder(tf.float32, shape=None)
            feed_dict = fill_feed_dict(data_set, images_u, images_v, labels_pl, prop, phase_train, 'test')
            raw_output, output, label = sess.run([logits, eval, labels_pl], feed_dict=feed_dict)
            raw_output_mat.append(raw_output)

            output_f = output[1][:,0].astype(np.float32)
            disp = (output_f * disp_precision) + disp_min + disp_precision / 2
            disp = disp.reshape([test_batch])
            true_disp = abs(disp - label) < 0.07
            true_count += np.sum(true_disp)
            output_txt.append(disp)

        precision = float(true_count) / num_example
        print time.time() - start
        print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))
        count += 1
        output_txt = np.array(output_txt)
        output_txt = output_txt.reshape([512, 512])
        name = data_set.get_data_name()
        np.savetxt(main_path + '/image_disp/output/' + name + '.txt', output_txt, fmt='%.5f')
        raw_output_mat = np.array(raw_output_mat)
        raw_output_mat = raw_output_mat.reshape([512, 512, class_num])
        sio.savemat(main_path + '/image_disp/output/' + name + '.mat', {'raw_output': raw_output_mat})


def fill_feed_dict(data_sets, images_u_pl, images_v_pl, labels_placeholder, prop_placeholder, phase_train,
                   mode='train'):
    count = batch_size
    if mode == 'test':
        count = test_batch
    images_u, images_v, labels = data_sets.next_batch(count)
    if mode == 'test':
        feed_dict = {
            images_u_pl: images_u,
            images_v_pl: images_v,
            labels_placeholder: labels,
            prop_placeholder: 1.0,
            phase_train: False,
        }
    elif mode == 'train':
        feed_dict = {
            images_u_pl: images_u,
            images_v_pl: images_v,
            labels_placeholder: labels,
            prop_placeholder: 0.5,
            phase_train: True,
        }
    return feed_dict


def main():
    print "initial model generator"
    with tf.Graph().as_default():
        train_sets = dataset.get_datasets(main_path, EPIWidth, disp_precision, 'train')
        test_sets = dataset.get_datasets(main_path, EPIWidth, disp_precision, 'test')

        global_step = tf.Variable(0, trainable=False)

        images_placeholder_v = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 1))
        images_placeholder_u = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 1))
        labels_placeholder = tf.placeholder(tf.int32, shape=None)
        prop_placeholder = tf.placeholder('float')
        phase_train = tf.placeholder(tf.bool, name='phase_train')

        logits = network.inference_ds(images_placeholder_u, images_placeholder_v, prop_placeholder, phase_train,
                                      disp_precision)

        logits_softmax = network.softmax(logits)

        loss = network.loss(logits_softmax, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        eval = network.evaluation(logits_softmax)

        summary = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())

        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))

        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
            #            saver.restore(sess,checkpoint_path+'/model.ckpt')#利用不同平台的训练结果
#            saver.restore(sess, ckpt.model_checkpoint_path)  # 本地训练的结果
            print ("restore from checkpoint!")
        else:
            print("no checkpoint found!")

        start_time = time.time()

        step = 0

        while not train_sets.complete:
            feed_dict = fill_feed_dict(train_sets, images_placeholder_u, images_placeholder_v, labels_placeholder,
                                       prop_placeholder, phase_train, 'train')
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time
            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 25000 == 24999:
                print('test Data Eval:')
                do_eval_true(sess, eval, logits_softmax, images_placeholder_u, images_placeholder_v, prop_placeholder,
                             phase_train, test_sets)

            if step % 50000 == 49999:
                saver.save(sess, checkpoint_path + '/model.ckpt', global_step=step)


def trans(x):
    r = int((x - disp_min) / disp_precision)
    if r < 0:
        r = 0
    if r > class_num:
        r = class_num
    return r

if __name__ == '__main__':
    main()

    print 'done'
