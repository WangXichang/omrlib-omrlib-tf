# *_* utf-8 *_*

import omr_lib1 as omr1ib1
import time
import os
import numpy as np
import cv2
import tensorflow as tf
import xml.etree.ElementTree as EleTree
import matplotlib.image as mg


def make_omr2018():

    import form_test as ftt
    # former = ftt.form2_omr01()    # omrimage2-1 omr01.jpg
    former = ftt.form2_OMR01()      # omrimage2-2 OMR01.jpg
    omrmodel = omr1ib1.OmrModel()

    omrxml = OmrDataset()
    omrxml.set_model(omrmodel=omrmodel, omrformer=former)
    omrxml.save_image_file = 'd:/study/dataset/omr2018b/JPEGImages/?'
    omrxml.save_xml_file = 'd:/study/dataset/omr2018b/Annotations/?'

    if not os.path.isdir(omrxml.save_xml_file.replace('?', '')):
        os.makedirs(omrxml.save_xml_file.replace('?', ''))
    if not os.path.isdir(omrxml.save_image_file.replace('?', '')):
        os.makedirs(omrxml.save_image_file.replace('?', ''))
    omrxml.create_dataset()


class OmrDataset(object):
    def __init__(self):
        self.omrmodel = omr1ib1.OmrModel()
        self.omrformer = None
        self.save_image_file = 'd:/study/dataset/omr2018/JPEGImages/?'
        self.save_xml_file = 'd:/study/dataset/omr2018/Annotations/?'
        self.omr_xml_tree = EleTree.parse('omr2018_annotation_temp.xml')
        self.root = self.omr_xml_tree.getroot()

    def set_model(self, omrmodel, omrformer):
        self.omrmodel = omrmodel
        self.omrformer = omrformer

    def create_dataset(self):
        for i, f in enumerate(self.omrformer.form['image_file_list']):
            xmlstr = EleTree.tostring(self.root)
            rt = omr1ib1.omr_test(self.omrformer, f)

            # save image file
            save_image_file_name = self.save_image_file.replace('?', '%05d.jpg' % i)
            if 1 == 1:
                mg.imsave(save_image_file_name, rt.image_card_2dmatrix)

            # xml content processing
            # xml--folder
            folder = self.save_image_file.split('/')[-3]
            xmlstr = xmlstr.replace(b'pppp', bytes(folder, encoding='utf8'))
            # xml--filename
            xmlstr = xmlstr.replace(b'xxxx.jpg',
                                    bytes(omr1ib1.Tools.find_file(save_image_file_name), encoding='utf8'))
            # xml--image size
            xmlstr = xmlstr.replace(b'image_size_width', bytes(str(rt.image_card_2dmatrix.shape[1]), encoding='utf8'))
            xmlstr = xmlstr.replace(b'image_size_height', bytes(str(rt.image_card_2dmatrix.shape[0]), encoding='utf8'))
            # xml--h_mark
            h_mark_xmin = str(rt.pos_xy_start_end_list[0][0])
            h_mark_ymin = str(rt.pos_xy_start_end_list[2][-1])
            h_mark_xmax = str(rt.pos_xy_start_end_list[1][-1])
            h_mark_ymax = str(rt.pos_xy_start_end_list[3][-1])
            xmlstr = xmlstr.replace(b'h_mark_xmin', bytes(h_mark_xmin, encoding='utf8'))
            xmlstr = xmlstr.replace(b'h_mark_ymin', bytes(h_mark_ymin, encoding='utf8'))
            xmlstr = xmlstr.replace(b'h_mark_xmax', bytes(h_mark_xmax, encoding='utf8'))
            xmlstr = xmlstr.replace(b'h_mark_ymax', bytes(h_mark_ymax, encoding='utf8'))
            # xml--v_mark
            v_mark_xmin = str(rt.pos_xy_start_end_list[0][-1])
            v_mark_ymin = str(rt.pos_xy_start_end_list[2][0])
            v_mark_xmax = str(rt.pos_xy_start_end_list[1][-1])
            v_mark_ymax = str(rt.pos_xy_start_end_list[3][-1])
            xmlstr = xmlstr.replace(b'v_mark_xmin', bytes(v_mark_xmin, encoding='utf8'))
            xmlstr = xmlstr.replace(b'v_mark_ymin', bytes(v_mark_ymin, encoding='utf8'))
            xmlstr = xmlstr.replace(b'v_mark_xmax', bytes(v_mark_xmax, encoding='utf8'))
            xmlstr = xmlstr.replace(b'v_mark_ymax', bytes(v_mark_ymax, encoding='utf8'))

            # xml--save annotaton.xml
            if 2 == 2:
                f = open(self.save_xml_file.replace('?', '%05d.xml' % i), 'w')
                f.write(xmlstr.decode(encoding='utf8'))
                f.close()


def set_omr_form(image_file_list=(),
                 mark_format=None,
                 group_format=None
                 ):
    """
    :param image_file_list: omr image files names list
    :param mark_format: omr recognition mark format dict
        {'mark_col_number':int,
         'mark_row_number':int,
         'mark_valid_area_col_start':int,
         'mark_valid_area_col_end':int,
         'mark_valid_area_row_start':int,
         'mark_valid_area_row_end':int}
    :param group_format: omr code group format dict
        {group_no,int:[(x,y),int,C,str,C ] #start_coordination:(int,int),length, direction:H/V, code, choice:M/S
         ......}
    :return: omr_form
    ---Example
        omr_image_location = 'C:/omrimage/*.jpg'
        omr_image_list = glob.glob(omr_image_location)
        group1 = {j: [(1, 23+j-1), 10, 'V', '0123456789', 'S'] for j in range(1, 15)}
        card_form = {
            'image_file_list': omr_image_list,
            'mark_format': {
                'mark_col_number': 37,
                'mark_row_number': 14,
                'mark_valid_area_col_start': 23,
                'mark_valid_area_col_end': 36,
                'mark_valid_area_row_start': 1,
                'mark_valid_area_row_end': 13},
                'group_format': group1
                }
    """
    omr_form = dict()
    omr_form['image_file_list'] = image_file_list
    omr_form['mark_format'] = mark_format
    omr_form['group_format'] = group_format
    return omr_form


def omr_save_tfrecord(card_form,
                      write_tf_file='tf_data',
                      image_reshape=(12, 16)):
    write_name = write_tf_file
    omr = omr1ib1.OmrModel()
    omr.set_mark_format(tuple([s for s in card_form['mark_format'].values()]))
    omr.set_group(card_form['group_format'])
    omr_writer = OmrTfrecordWriter(write_name,
                                   image_reshape=image_reshape)
    sttime = time.clock()
    run_len = len(card_form['image_file_list'])
    run_len = run_len if run_len > 0 else -1
    pbar = omr1ib1.ProgressBar(0, run_len)
    run_count = 0
    for f in card_form['image_file_list']:
        omr.set_omr_image_filename(f)
        omr_writer.read_omr_write_tfrecord(omr)
        run_count += 1
        pbar.move()
        pbar.log(f)
    del omr_writer
    total_time = round(time.clock()-sttime, 2)
    print(f'total_time={total_time}  mean_time={round(total_time / run_count, 2)}')
    return run_len


class OmrTfrecordWriter:
    """
    write tfrecords file batch
    function:
        save omr_dict to tfrecord file{features:label, painting_block_image}
    parameters:
        param tfr_pathfile: lcoation+filenamet to save omr label + blockimage
        param image_reshape, resize iamge to the shape
    return:
        TFRecord file= [tfr_pathfile].tfrecord
    """

    def __init__(self, tfr_pathfile, image_reshape=(12, 16)):
        self.tfr_pathfile = tfr_pathfile
        self.image_reshape = image_reshape
        # self.sess = tf.Session()
        self.writer = tf.python_io.TFRecordWriter(tfr_pathfile)

    def __del__(self):
        # self.sess.close()
        self.writer.close()

    def read_omr_write_tfrecord(self, omr):
        old_status = omr.debug
        omr.debug = True
        omr.run()
        # df = omr.get_result_dataframe2()
        df = omr.omr_result_dataframe
        omr.debug = old_status
        data_labels = df[df.group > 0][['coord', 'label']].values
        data_images = omr.omrdict
        # st = time.clock()
        self.write_tfrecord(data_labels, data_images)
        # print(f'{omr.image_filename}  consume time={time.clock()-st}')

    def write_tfrecord(self, dataset_labels: list, dataset_images: dict):
        # param dataset_labels: key(coord)+label(str), omr block image label ('0'-unpainted, '1'-painted)
        # param dataset_images: key(coord):blockiamge
        for key, label in dataset_labels:
            omr_image = dataset_images[key]
            omr_image3 = omr_image.reshape([omr_image.shape[0], omr_image.shape[1], 1])
            resized_image = cv2.resize(omr_image3,
                                       (self.image_reshape[1], self.image_reshape[0]),
                                       cv2.INTER_NEAREST)
            # resized_image = cv2.resize(omr_image3.astype('float'),
            #                           self.image_reshape, cv2.INTER_NEAREST).astype('int')
            # resized_image = tf.image.resize_images(omr_image3, self.image_reshape)
            # resized_image = omr_image
            # bytes_image = self.sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            bytes_image = resized_image.tobytes()
            if type(label) == int:
                label = str(label)
            omr_label = label.encode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[omr_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))
            }))
            self.writer.write(example.SerializeToString())


class OmrTfrecordIO:
    """
    processing data with tensorflow
    """

    @staticmethod
    def fun_save_omr_tfrecord(tfr_pathfile: str, dataset_labels: list, dataset_images: dict,
                              image_shape=(10, 15)):
        """
        function:
            save omr_dict to tfrecord file{features:label, painting_block_image}
        parameters:
            param tfr_pathfile: lcoation+filenamet to save omr label+blockimage
            param dataset_labels: key(coord)+label(str), omr block image label ('0'-unpainted, '1'-painted)
            param dataset_images: key(coord):blockiamge
        return:
            TFRecord file= [tfr_pathfile].tfrecord
        """
        st = time.clock()
        sess = tf.Session()
        writer = tf.python_io.TFRecordWriter(tfr_pathfile+'.tfrecord')
        for key, label in dataset_labels:
            omr_image = dataset_images[key]
            omr_image3 = omr_image.reshape([omr_image.shape[0], omr_image.shape[1], 1])
            resized_image = tf.image.resize_images(omr_image3, image_shape)
            # resized_image = omr_image
            bytes_image = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            if type(label) == int:
                label = str(label)
            omr_label = label.encode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[omr_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        sess.close()
        print(f'consume time={time.clock()-st}')

    @staticmethod
    def fun_read_tfrecord_queue_sess(tfr_pathfile,
                                     shuffle_batch_size=30,
                                     shuffle_capacity=2000,
                                     shuffle_min_after_dequeue=1000,
                                     image_shape=(12, 16, 1)
                                     ):
        # create queue to read data from tfrecord file
        # file_name = 'tf_card_1.tfrecords'  #tfr_pathfiles
        # image, label = OmrTfrecordIO.fun_read_tfrecord_queue(tfr_pathfiles=file_name)
        image_resize_shape = image_shape    # (10, 15, 1)
        # omr_data_queue = tf.train.string_input_producer(
        #                    tf.train.match_filenames_once(file_name))
        omr_data_queue = tf.train.string_input_producer([tfr_pathfile])
        reader = tf.TFRecordReader()
        _, ser = reader.read(omr_data_queue)
        omr_data = tf.parse_single_example(
            ser,
            features={
                'label': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature([], tf.string)
            }
        )
        omr_image = tf.decode_raw(omr_data['image'], tf.uint8)
        image = tf.reshape(omr_image, image_resize_shape)
        label = tf.cast(omr_data['label'], tf.string)
        # 使用shuffle_batch可以随机打乱输入
        img_batch, label_batch = \
            tf.train.shuffle_batch([image, label],
                                   batch_size=shuffle_batch_size,
                                   capacity=shuffle_capacity,
                                   min_after_dequeue=shuffle_min_after_dequeue)
        # init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.initialize_all_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(10):
                val, la = sess.run([img_batch, label_batch])
                # 也可以根据需要对val， l进行处理
                # l = to_categorical(l, 12)
                print(val.shape, la)
            coord.request_stop()
            coord.join(threads)

    @staticmethod
    def fun_read_tfrecord_queue(tfr_pathfiles,
                                image_shape=(12, 16, 1)):
        redict = dict()
        redict[0] = []
        redict[1] = []
        # image_resize
        image_resize_shape = image_shape  # [10, 15, 1]
        omr_data_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(tfr_pathfiles)
        )
        with tf.TFRecordReader() as reader:
            _, ser = reader.read(omr_data_queue)
            omr_data = tf.parse_single_example(
                ser,
                features={
                    'label': tf.FixedLenFeature([], tf.string),
                    'image': tf.FixedLenFeature([], tf.string),
                })
            omr_image = tf.decode_raw(omr_data['image'], tf.uint8)
            omr_image_reshape = tf.reshape(omr_image, image_resize_shape)
            omr_label = tf.cast(omr_data['label'], tf.string)
            redict[0].append(omr_image_reshape)
            redict[1].append((1, 0) if tf.equal(omr_label, tf.constant(0)) else (0, 1))
        # return omr_image_reshape, omr_label
        return omr_image_reshape, omr_label

    @staticmethod
    def fun_read_tfrecord_tolist(tfr_file, image_shape=(12, 16), readnum=2000):
        # no session method
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
            # print(len(image[0]))
            # 做一些预处理
            img = np.zeros([image_shape[0] * image_shape[1]])
            for i in range(len(img)):
                img[i] = image[0][i]
            # for i in range(image_shape[0]):
            #    for j in range(image_shape[1]):
            #        img[i, j] = image[0][i*image_shape[1] + j]
            image_list.append(img)
            labelvalue = int(chr(label[0][0]))
            label_list.append((1, 0) if labelvalue == 0 else (0, 1))
            count += 1
            if count >= readnum:
                break
            if count % 5000 == 0:
                print(count)
        return image_list, label_list

    @staticmethod
    def fun_read_tfrecord_filelist(file_list, read_num=100):
        """
        read label,image from tfrecord datafile
        read number is not limited, 0 - indefinite
        use tensorflow.Session
        :param file_list: [tfrecordfilename, ... ]
        :param read_num:  records number to read out
        :return:
            image_dict: {No:iamgematrix, ...}
            label_list: [labelvalue, ...]
        :note
            label_list index is corresponding to image_dict key
        """
        # check file list is ok
        if type(file_list) != list:
            if type(file_list) == str:
                file_list = [file_list]
            else:
                print('need a file_name or files_name_list!')
                return
        for s in file_list:
            if not os.path.isfile(s):
                print(f'file error: not found file:\"{s}\"!')
                return {}, []
        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer(file_list)
        reader = tf.TFRecordReader()
        # 返回文件名和文件
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'image': tf.FixedLenFeature([], tf.string),
                                           })
        label = tf.cast(features['label'], tf.string)
        image = tf.decode_raw(features['image'], tf.uint8)
        # img = tf.reshape(img, [10, 15, 1])
        # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        image_dict = {}
        label_list = []
        with tf.Session() as sess:
            # init_op = tf.initialize_all_variables()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(read_num):
                ex_image, ex_label = sess.run([image, label])  # 取出image和label
                # img=Image.fromarray(example.reshape([15, 12]), 'L')  # Image from PIL
                # img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')  #存下图片
                # print(i, ex_image.shape, ex_label)
                image_dict[i] = ex_image
                label_list.append(int(chr(ex_label[0])))
            coord.request_stop()
            coord.join(threads)
        return image_dict, label_list
