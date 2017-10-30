import tensorflow as tf


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
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # label = tf.cast(features['label'], tf.int32)
    lab = features['label']
    return img, lab


image, label = read_and_decode("test_card2.tfrecord", [12, 15, 1])
# 使用shuffle_batch可以随机打乱输入
image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                  batch_size=30, capacity=2000,
                                                  min_after_dequeue=1000)
init = tf.global_variables_initializer()  # tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(10):
        values, labels = sess.run([image_batch, label_batch])
        # 根据需要对val， labels进行处理
        # l = to_categorical(l, 12)
        print(values.shape)
        print(labels)
        # print([int(values[j].mean()) for j in range(30)])
        # print([int(chr(v[0])) for v in labels])
