# _*_ utf-8 _*_

import tensorflow as tf
import omr_lib2 as omr2
import os
import numpy as np


def exp_data():
    # get omr data
    data = Data()
    data.tf_data_file = data.office_card1
    data.read_data()
    return data


def exp_model(data):
    # training model
    model = CnnModel()
    model.save_model_path_name = model.office_model_path + model.model_name
    model.train_model(data)
    return model


def exp_model_test(model_path_name, dataset):
    # use model
    model_app = CnnApp()
    model_app.set_model(model_path_name)
    model_app.load_model()
    model_app.test(dataset)
    model_app.predict(dataset[0][0:10])
    return model_app


def exp_model_predict(model_path_name, image_data):
    # use model
    model_app = CnnApp()
    model_app.set_model(model_path_name)
    model_app.load_model()
    model_app.predict(image_data[0][0:10])
    return model_app


class Data:
    def __init__(self):
        self.office_card1 = 'f:/studies/juyunxia/tfdata/tf_card1_1216.tfrecords'
        self.office_card2 = 'f:/studies/juyunxia/tfdata/tf_card2_1216.tfrecords'
        self.office_card3 = 'f:/studies/juyunxia/tfdata/tf_card3_1216.tfrecords'
        self.dell_card1 = 'd:/work/data/omr_tfrecords/tf_card1_1216.tfrecords'
        # data file to read
        self.tf_data_file = None
        self.example_len = 1000
        self.train_rate = 0.85
        self.train_len = int(self.example_len * self.train_rate)
        self.test_len = self.example_len - self.train_len
        self.image_shape = (12, 16)
        self.data_set = None  # [image_array, label_array]  # iamge data normlized by 1/255, label one-hot

    def read_data(self):
        self.data_set = \
            self.fun_read_tfrecord_tolist(self.tf_data_file)
        self.example_len = len(self.data_set[0])
        self.train_len = int(self.example_len * self.train_rate)
        self.test_len = self.example_len - self.train_len

    def fun_read_tfrecord_tolist(self, tf_data_file):
        # get image, label data from tfrecord file
        # labellist = [lableitems:int, ... ]
        # imagedict = [label:imagematix, ...]
        if not os.path.isfile(tf_data_file):
            print(f'file error: not found file: \"{tf_data_file}\"!')
            return {}, []
        count = 0
        image_list = []
        label_list = []
        example = tf.train.Example()
        for serialized_example in tf.python_io.tf_record_iterator(tf_data_file):
            example.ParseFromString(serialized_example)
            image = example.features.feature['image'].bytes_list.value
            label = example.features.feature['label'].bytes_list.value
            # precessing
            # img = np.zeros([image_shape[0] * image_shape[1]])
            # for i in range(len(img)):
            #    img[i] = image[0][i]
            img = np.array([image[0][x] for x in range(len(image[0]))])
            image_list.append(img / 255)
            labelvalue = int(chr(label[0][0]))
            label_list.append((1, 0) if labelvalue == 0 else (0, 1))
            count += 1
            if count % 3000 == 0:
                print(count)
        print(f'total images= {count}')
        return image_list, label_list

    def get_train_data(self, batchnum, starting_location):
        # read batchnum data in [0, train_len]
        if batchnum > self.train_len:
            batchnum = self.train_len
        if (starting_location + batchnum) > self.train_len:
            starting_location = 0
        # print(f'train examples {callnum}')
        res_data = [[self.data_set[0][i] for i in range(starting_location, starting_location + batchnum)],
                    self.data_set[1][starting_location:starting_location + batchnum]]
        return res_data

    def get_test_data(self):
        res_data = [[self.data_set[0][i]
                     for i in range(self.train_len, self.train_len+self.test_len)],
                    self.data_set[1][self.train_len:self.train_len + self.test_len]]
        return res_data


class CnnModel:

    def __init__(self):
        self.office_model_path = 'f:/studies/juyunxia/omrmodel/'
        self.dell_model_path = 'd:/work/omrmodel/'
        self.model_name = 'omr_model'
        self.save_model_path_name = './' + self.model_name

    def weight_var(self, shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_value=init)

    def bias_var(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=init)

    def conv2d(self, x, w):
        res = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        return res

    def max_pool_2x2(self, x):
        res = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
        return res

    def train_model(self, data: Data, batch_num=40, train_num=1000):
        """ use dataset to train model"""

        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, [None, 192], name='input_omr_images')
        y_ = tf.placeholder(tf.float32, [None, 2], name='input_omr_labels')
        x_image = tf.reshape(x, [-1, 12, 16, 1])

        w_conv1 = self.weight_var([4, 6, 1, 32])
        b_conv1 = self.bias_var([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        w_conv2 = self.weight_var([4, 4, 32, 128])
        b_conv2 = self.bias_var([128])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        w_fc1 = self.weight_var([3*4*128, 256])
        b_fc1 = self.bias_var([256])
        h_pool1_flat = tf.reshape(h_pool2, [-1, 3*4*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = self.weight_var([256, 2])
        b_fc2 = self.bias_var([2])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                                                      reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        tf.add_to_collection('predict_label', y_conv)
        tf.add_to_collection('accuracy', accuracy)

        print(f'data={data.tf_data_file}')
        for i in range(train_num):
            batch = data.get_train_data(batch_num, i * 20)
            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0
                })
                print("step=%4d,\t accuracy= %2.8f" % (i, train_accuracy), '\t',
                      'cross_entropy=%1.10f' % cross_entropy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                      )
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

        omr_test_data = data.get_test_data()
        print('test accuracy= %1.6f' % accuracy.eval(
            feed_dict={
                x: omr_test_data[0], y_: omr_test_data[1], keep_prob: 1.0
            }
        ))
        saver.save(sess, self.save_model_path_name+'.ckpt')

    def use_model(self, omr_image_data):
        # saver = tf.train.Saver()
        modelmeta = self.save_model_path_name + '.ckpt.meta'
        modelckpt = self.save_model_path_name + '.ckpt'
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(modelmeta)
        with tf.Session() as sess:
            saver.restore(sess, modelckpt)
            y = tf.get_collection('predict_label')[0]
            a = tf.get_collection('accuracy')[0]
            graph = tf.get_default_graph()
            # y 有placeholder "input_omr_images"，
            # sess.run(y)的时候还需要用实际待预测的样本
            # 以及相应的参数(keep_porb)来填充这些placeholder，
            # 这些需要通过graph的get_operation_by_name方法来获取。
            input_x = graph.get_operation_by_name('input_omr_images').outputs[0]
            input_y = graph.get_operation_by_name('input_omr_labels').outputs[0]
            keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
            # 使用 y 进行预测
            yp = sess.run(y, feed_dict={input_x: omr_image_data[0], keep_prob: 1.0})
            ac = sess.run(a, feed_dict={input_x: omr_image_data[0],
                                        input_y: omr_image_data[1],
                                        keep_prob: 1.0})
        return yp, ac


class CnnApp:

    def __init__(self):
        self.model_path_name = 'f:/studies/juyunxia/omrmodel/omr_model'
        self.sess = tf.Session()
        self.saver = None
        self.graph = None
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.y = None
        self.a = None

    def __del__(self):
        self.sess.close()

    def set_model(self, modelstr: str):
        self.model_path_name = modelstr

    def load_model(self):
        tf.reset_default_graph()
        self.saver = tf.train.import_meta_graph(self.model_path_name + '.ckpt.meta')
        self.saver.restore(self.sess, self.model_path_name+'.ckpt')
        self.y = tf.get_collection('predict_label')[0]
        self.a = tf.get_collection('accuracy')[0]
        self.graph = tf.get_default_graph()
        self.input_x = self.graph.get_operation_by_name('input_omr_images').outputs[0]
        self.input_y = self.graph.get_operation_by_name('input_omr_labels').outputs[0]
        self.keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]

    def test(self, omr_data_set):
        # 预测, 计算识别率
        yp = self.sess.run(self.y, feed_dict={self.input_x: omr_data_set[0], self.keep_prob: 1.0})
        ac = self.sess.run(self.a, feed_dict={self.input_x: omr_data_set[0],
                                              self.input_y: omr_data_set[1],
                                              self.keep_prob: 1.0})
        return yp, ac

    def predict(self, omr_image_set):
        # 使用 y 进行预测
        yp = self.sess.run(self.y, feed_dict={self.input_x: omr_image_set, self.keep_prob: 1.0})
        return yp

