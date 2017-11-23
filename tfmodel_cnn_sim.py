# _*_ utf-8 _*_

import tensorflow as tf
import omr_lib2 as omr2


fs = 'c:/users/wangxichang/students/ju/testdata/tfrecord_data/tf_card1_1216.tfrecords'
batch_images_all = omr2.OmrTfrecordIO.fun_read_tfrecord_tolist(fs, [12, 16])
train_len = 29000
test_len = 400
callnum = 0

sess = tf.InteractiveSession()

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

def get_omr_images(batchnum, callnum, maxlen = train_len):
    # read omr batchnum images
    # exercise:
    ''' for test
    import numpy as np
    res_data = [0, 1]
    res_data[0] = [np.random.randint(0, 255, [192])
                   for _ in range(batchnum)]
    res_data[1] = [np.random.randint(0, 2, [2])
                   for _ in range(batchnum)]
    '''
    if callnum > (maxlen - batchnum):
        callnum = 0
    # print(f'train examples {callnum}')
    res_data = [batch_images_all[0][callnum:callnum+batchnum],
                batch_images_all[1][callnum:callnum+batchnum]]
    return res_data

def get_omr_test_images():
    res_data = [batch_images_all[0][29300:29400],
                batch_images_all[1][29300:29400]]
    return res_data

x = tf.placeholder(tf.float32, [None, 192])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x, [-1, 12, 16, 1])

w_conv1 = weight_var([4, 4, 1, 32])
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_fc1 = weight_var([6*8*32, 256])
b_fc1 = bias_var([256])
h_pool1_flat = tf.reshape(h_pool1, [-1, 6*8*32])
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
for i in range(500):
    batch = get_omr_images(30, i*30)
    if i % 20 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.3
        })
        print("step %d, accuracy= %2.4f" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

omr_test_data = get_omr_test_images()
print('test accuracy= %g' % accuracy.eval(
    feed_dict={
        x:omr_test_data[0], y_:omr_test_data[1], keep_prob:1.0
    }
))
