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

def do_eval_true(sess, eval, logits, images_u, image_v, prop, phase_train, data_set):
    count = 0
    while count < data_set.num_of_path:
        true_count = 0
        raw_output_mat = []
        output_txt = []
        steps_per_epoch = data_set.num_examples // test_batch
        num_example = steps_per_epoch * test_batch
        for step in xrange(steps_per_epoch):
            labels_pl = tf.placeholder(tf.float32, shape=None)
            feed_dict = fill_feed_dict(data_set,images_u,image_v,labels_pl,prop,phase_train)
            raw_output, output, label = sess.run([logits, eval, labels_pl],feed_dict=feed_dict)
            raw_output_mat.append(raw_output)
            for i in xrange(test_batch):
                disp = (output[1][i]*disp_precision)-2+disp_precision/2
                if disp>2:
                    disp=2.0
                true_count += abs(disp-label[i])<0.07
                output_txt.append(disp)
        precision = float(true_count) / num_example
        print ('example: %d, correct: %d, Precision: %0.04f' % (num_example, true_count, precision))
        count += 1
        output_txt = np.array(output_txt)
        output_txt = output_txt.reshape([512,512])
        raw_output_mat = np.array(raw_output_mat)
        raw_output_mat = raw_output_mat.reshape([512,512,58])
        name = data_set.get_data_name()
        np.savetxt(main_path+'/image_disp/'+name+'.txt',output_txt,fmt='%.5f')
        np.save(main_path+'/image_disp/'+name+'.npy',raw_output_mat)
        sio.savemat(main_path+'/image_disp/'+name+'.mat',{'raw_output':raw_output_mat})


def fill_feed_dict(data_sets, images_u_pl, image_v_pl, labels_placeholder, prop_placeholder,phase_train):
    images_u, images_v, labels = data_sets.next_batch(test_batch)
    feed_dict = {
        images_u_pl: images_u,
        image_v_pl: images_v,
        labels_placeholder: labels,
        prop_placeholder:1.0,
        phase_train:False,
    }
    return feed_dict


def main():
    with tf.Graph().as_default():
        test_sets = dataset.get_datasets(main_path, EPIWidth, disp_precision, 'test')

        images_placeholder_v = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 3))
        images_placeholder_u = tf.placeholder(tf.float32, shape=(None, 9, EPIWidth, 3))
        prop_placeholder = tf.placeholder('float')
        phase_train = tf.placeholder(tf.bool,name='phase_train')

        logits = network.inference_ds(images_placeholder_u,images_placeholder_v,prop_placeholder,phase_train,EPIWidth,disp_precision)

        eval = network.evaluation(logits)

        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
#            saver.restore(sess,checkpoint_path+'/model.ckpt')#利用不同平台的训练结果
            saver.restore(sess,ckpt.model_checkpoint_path)#本地训练的结果
            print ("restore from checkpoint!")
        else:
            print("no checkpoint found!")

        print('Training Data Eval:')
        do_eval_true(sess,eval,logits,images_placeholder_u,images_placeholder_v,prop_placeholder,phase_train,test_sets)



if __name__ == '__main__':
    main()
    str = 'abc/def'
    str = str[str.find('/')+1:]
    print str


