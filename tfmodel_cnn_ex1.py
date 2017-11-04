import tensorflow as tf
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')


def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')


def conv_2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def evaluate(y, y_):
    y = tf.argmax(input=y)  # , dimension=1)  # tf.arg_max(input=y, dimension=1)
    y_ = tf.argmax(input=y_)  # , dimension=1)  #tf.arg_max(input=y_, dimension=1)
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32))


def read_and_decode(filename, image_reshape=(10, 15, 1)):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, image_reshape)  # [10, 15, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    #lab = tf.cast(features['label'], tf.int32)
    lab = features['label']
    return img, lab


def test_cnn(batch_size=30, lr=0.0001, num_iter=20000):
    # prepare omr dataset
    # dataset = input_data.read_data_sets(train_dir='MNIST_data/', one_hot=True)
    # image_test, label_test = read_and_decode("tf_data.tfrecords")
    image, label = read_and_decode("tf_card_2.tfrecords", [10, 15, 1])
    '''
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              capacity= 300)
    '''
    # 使用shuffle_batch可以随机打乱输入
    # image_batch, label_batch = tf.train.shuffle_batch([image, label],
    #                                                   batch_size=batch_size,
    #                                                   capacity=200,
    #                                                  min_after_dequeue=100)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 150], name='images')
    y = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='labels')
    # 后面的卷积操作输入参数必须为‘float32’或者‘float64’

    w_conv1 = weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = bias_variable(shape=[32])
    reshape_x = tf.reshape(x, shape=[-1, 28, 28, 1])  # 省略的形式区别于占位符!!!!!!
    conv1_out = tf.nn.relu(conv_2d(reshape_x, w_conv1)+b_conv1)
    pool1_out = max_pool_2x2(conv1_out)

    w_conv2 = weight_variable(shape=[5, 5, 32, 64])
    b_conv2 = bias_variable(shape=[64])
    conv2_out = tf.nn.relu(conv_2d(pool1_out, w_conv2) + b_conv2)
    pool2_out = max_pool_2x2(conv2_out)

    full_connected_in = tf.reshape(pool2_out, shape=[-1, 7*7*64])
    w_full_connected = weight_variable(shape=[7*7*64, 1024])
    b_full_connected = bias_variable(shape=[1024])
    full_connected_out1 = tf.nn.relu(tf.matmul(full_connected_in, w_full_connected)+b_full_connected)
    # drop out防止过拟合
    dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')
    full_connected_out = tf.nn.dropout(x=full_connected_out1, keep_prob=dropout_prob)

    w_softmax = weight_variable(shape=[1024, 2])
    b_softmax = bias_variable(shape=[2])
    softmax_in = tf.matmul(full_connected_out, w_softmax) + b_softmax
    softmax_out = tf.nn.softmax(logits=softmax_in, name='softmax_layer')
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_in, labels=y)
    step_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss)
    # 在测试数据集上评估算法的准确率
    accuracy = evaluate(y, softmax_out)
    # initialized_variables = tf.initialize_all_variables()
    initialized_variables = tf.global_variables_initializer()

    print('Start to train the convolutional neural network......')
    sess = tf.Session()
    sess.run(fetches=initialized_variables)
    for iter in range(num_iter):
        # image, label = read_and_decode("tf_card_2.tfrecords", [10, 15, 1])
        # batch = dataset.train.next_batch(batch_size=batch_size)
        # sess.run(fetches=Step_train, feed_dict={x: batch[0], y: batch[1], dropout_prob: 0.5})
        x_image, y_label = sess.run([image, label])
        sess.run(fetches=step_train, feed_dict={x: x_image, y: y_label, dropout_prob: 0.5})
        if (iter+1) % 100 == 0:  # 计算在当前训练块上的准确率
            accuracy_id = sess.run(fetches=accuracy, feed_dict={x: x_image,
                                                             y: y_label, dropout_prob: 1})
            print('Iter num %d ,the train accuracy is %.3f' % (iter+1, accuracy_id))

    #Accuracy = sess.run(fetches=accuracy, feed_dict={x: image_test,
    #                                                 y: label_test,
    #                                                 dropout_prob: 1})
    sess.close()
    # print('Train process finished, the best accuracy is %.3f' % Accuracy)


if __name__ == '__main__':
    test_cnn()
