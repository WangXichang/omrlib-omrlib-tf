# _*_ utf-8 _*_

import tensorflow as tf
import omr_lib2 as omr2
import os
import numpy as np


class OmrData:
    def __init__(self):
        # fs = 'c:/users/wangxichang/students/ju/testdata/tfrecord_data/tf_card3_1216.tfrecords'
        self.data_set = None
        self.tf_data_file = 'd:/work/data/omr_tfrecords/tf_card1_1216.tfrecords'
        self.example_len = 1000
        self.train_rate = 0.85
        self.train_len = int(self.example_len * self.train_rate)
        self.test_len = self.example_len - self.train_len
        self.image_shape = (12, 16)

    def read_data(self):
        self.data_set = \
            self.fun_read_tfrecord_tolist(self.tf_data_file)
        self.example_len = len(self.data_set[0])
        self.train_len = int(self.example_len * self.train_rate)
        self.test_len = self.example_len - self.train_len

    def fun_read_tfrecord_tolist(self, tfr_file):
        # get image, label data from tfrecord file
        # labellist = [lableitems:int, ... ]
        # imagedict = [label:imagematix, ...]
        if not os.path.isfile(tfr_file):
            print(f'file error: not found file: \"{tfr_file}\"!')
            return {}, []
        count = 0
        image_list = []
        label_list = []
        example = tf.train.Example()
        for serialized_example in tf.python_io.tf_record_iterator(tfr_file):
            example.ParseFromString(serialized_example)
            image = example.features.feature['image'].bytes_list.value
            label = example.features.feature['label'].bytes_list.value
            # precessing
            # img = np.zeros([image_shape[0] * image_shape[1]])
            #for i in range(len(img)):
            #    img[i] = image[0][i]
            img = np.array([image[0][x] for x in range(len(image[0]))])
            image_list.append(img)
            labelvalue = int(chr(label[0][0]))
            label_list.append((1, 0) if labelvalue == 0 else (0, 1))
            count += 1
            #if count >= readnum:
            #    break
            if count % 5000 == 0:
                print(count)
        return image_list, label_list

    def get_train_data(self, batchnum, starting_location):
        # read batchnum data in [0, train_len]
        if batchnum > self.train_len:
            batchnum = self.train_len
        if (starting_location + batchnum) > self.train_len:
            starting_location = 0
        # print(f'train examples {callnum}')
        res_data = [[self.data_set[0][i] / 255 for i in range(starting_location, starting_location + batchnum)],
                    self.data_set[1][starting_location:starting_location + batchnum]]
        return res_data

    def get_test_data(self):
        res_data = [[self.data_set[0][i] / 255
                     for i in range(self.train_len, self.train_len+self.test_len)],
                    self.data_set[1][self.train_len:self.train_len + self.test_len]]
        return res_data


def weight_var(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=init)


def bias_var(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=init)


def conv2d(x, w):
    res = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    return res


def max_pool_2x2(x):
    res = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')
    return res


def inference(data_set: OmrData):

    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 192])
    y_ = tf.placeholder(tf.float32, [None, 2])
    x_image = tf.reshape(x, [-1, 12, 16, 1])

    w_conv1 = weight_var([4, 6, 1, 32])
    b_conv1 = bias_var([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_var([4, 4, 32, 128])
    b_conv2 = bias_var([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_fc1 = weight_var([3*4*128, 256])
    b_fc1 = bias_var([256])
    h_pool1_flat = tf.reshape(h_pool2, [-1, 3*4*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_var([256, 2])
    b_fc2 = bias_var([2])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                  reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    print(f'data={data_set.tf_data_file}')
    for i in range(1000):
        batch = data_set.get_train_data(40, i * 20)
        if i % 50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
            })
            print("step=%4d,\t accuracy= %2.8f" % (i, train_accuracy), '\t',
                  'cross_entropy=%1.10f' % cross_entropy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                  )
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    omr_test_data = data_set.get_test_data()
    print('test accuracy= %1.6f' % accuracy.eval(
        feed_dict={
            x: omr_test_data[0], y_: omr_test_data[1], keep_prob: 1.0
        }
    ))
    saver.save(sess, 'd:/work/omrmodel/omr_model.ckpt')
