import tensorflow as tf

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [12, 15, 1])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # label = tf.cast(features['label'], tf.int32)
    label = features['label']

    return img, label

img, label = read_and_decode("test_card2.tfrecord")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()  #tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        print(val.shape, l)
        print(type(val))
