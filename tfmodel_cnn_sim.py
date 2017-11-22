# _*_ utf-8 _*_

import tensorflow as tf

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

def get_omr_images(batchnum):
    # read omr batchnum images
    # exercise:
    import numpy as np
    res_data = [0, 1]
    res_data[0] = [np.random.randint(0, 255, [192])
                   for _ in range(batchnum)]
    res_data[1] = [np.random.randint(0, 2, [2])
                   for _ in range(batchnum)]
    return res_data

def get_omr_test_images():
    return get_omr_images(100)

x = tf.placeholder(tf.float32, [None, 192])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x, [-1, 12, 16, 1])

w_conv1 = weight_var([4, 8, 1, 32])
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_fc1 = weight_var([6*8*32, 128])
b_fc1 = bias_var([128])
h_pool1_flat = tf.reshape(h_pool1, [-1, 6*8*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_var([128, 2])
b_fc2 = bias_var([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(1000):
    batch = get_omr_images(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.5
        })
        print("step %d, accuracy= %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

omr_test_data = get_omr_test_images()
print('test accuracy= %g' % accuracy.eval(
    feed_dict={
        x:omr_test_data[0], y_:omr_test_data[1], keep_prob:1.0
    }
))
