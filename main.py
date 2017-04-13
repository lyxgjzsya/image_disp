# -*- coding: UTF-8 -*-
import time
import tensorflow as tf
import network
import dataset


EPIWidth = 33
batch_size = 50
main_path = '/home/luoyaox/Work'
#main_path = '/home/cs505/workspace/luo_space'
box_path = main_path+'/box'
summary_path = main_path+'/image_disp/summary'
checkpoint_path = main_path+'/image_disp/checkpoint'
disp_precision = 0.07


def do_eval(sess, eval_correct, images, labels, prop, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_example = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images, labels, prop, mode='test')
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_example
    print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))

def do_eval_true(sess, eval, images_pl, prop, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size
    num_example = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        images, label = data_set.next_batch(batch_size)
        feed_dict = {
            images_pl:images,
            prop:1,
        }
        output = sess.run(eval,feed_dict=feed_dict)
        for i in xrange(batch_size):
            disp = (output[1][i]*disp_precision)-2+disp_precision/2
            if disp>2:
                disp=2
            true_count += abs(disp-label[i])<0.07

    precision = float(true_count) / num_example
    print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))


def fill_feed_dict(data_sets, images_placeholder, labels_placeholder, prop_placeholder, mode='train'):
    images, labels = data_sets.next_batch(batch_size)
    prop = 0.5
    if mode == 'test':
        prop = 1
    feed_dict = {
        images_placeholder: images,
        labels_placeholder: labels,
        prop_placeholder: prop,
    }
    return feed_dict


def main():
    train_sets, test_sets = dataset.get_datasets(box_path, EPIWidth, disp_precision)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_placeholder = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=None)
        prop_placeholder = tf.placeholder('float')


        logits = network.inference_old(images_placeholder, prop_placeholder, EPIWidth, disp_precision)

        loss = network.loss(logits, labels_placeholder)

        train_op = network.training(loss, 1e-4, global_step)

        eval_correct = network.evaluation(logits, labels_placeholder)

        evalv2 = network.evaluationv2(logits)

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

            feed_dict = fill_feed_dict(train_sets, images_placeholder, labels_placeholder, prop_placeholder)
            _, loss_value, output = sess.run([train_op, loss, logits], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 1000 == 0:
                print ('Step:%d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % 200 == 0:
                if step != 0:
                    saver.save(sess, checkpoint_path+'/model.ckpt',global_step=step)
                    print('Training Data Eval:')
#                    do_eval(sess, eval_correct, images_placeholder, labels_placeholder, prop_placeholder, test_sets)
                    do_eval_true(sess,evalv2,images_placeholder,prop_placeholder,test_sets)



if __name__ == '__main__':
    main()

