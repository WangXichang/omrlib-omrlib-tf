# *_* utf-8 *_*
# python 3.6x


import numpy as np
import pandas as pd
import matplotlib.image as mg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import os
import sys
from scipy.ndimage import filters
import copy
# import gc
# import tensorflow as tf
# import cv2
# from PIL import Image
# from sklearn import svm


def help_omr_model():
    print(OmrModel.__doc__)


def help_omr_form():
    print(OmrForm.__doc__)


def help_read_batch():
    print(omr_read_batch.__doc__)


def help_omr_():
    return OmrModel.omr_code_standard_dict


def omr_read_batch(card_form: dict,
                   result_save=False,
                   result_save_file='omr_data',
                   result_group=True
                   ):
    """
    :input
        card_form: form_dict, could get from class OmrForm
        result_save: bool, True=save result dataframe to disk file(result_save_file)
        result_save_file: file name to save data, auto added .csv
        result_group: bool, True, record error choice group info in field: group
    :return:
        omr_result_dataframe:
            card,   # card file name
            result, # recognized code string
            len,    # result string length
            group_result    # if result_group=True, group no for result delimited with comma, 'g1,g2,...,gn'
    """
    # mark_format = [v for v in card_form['mark_format'].values()]
    # group = card_form['group_format']

    if result_save:
        no = 1
        while os.path.isfile(result_save_file+'.csv'):
            result_save_file += '_'+str(no)
            no += 1
        result_save_file += '.csv'
        if not os.path.isdir(Tools.find_path(result_save_file)):
            print('invaild path in filename:' + result_save_file)
            return
    # set model
    omr = OmrModel()
    omr.set_form(card_form)
    omr.sys_group_result = result_group
    image_list = card_form['image_file_list']
    if len(image_list) == 0:
        print('no file found in card_form.image_file_list !')
        return None
    # run model
    omr_result = None
    sttime = time.clock()
    run_len = len(image_list)
    run_count = 0
    progress = ProgressBar(total=run_len)
    for f in image_list:
        omr.set_omr_image_filename(f)
        omr.run()
        rf = omr.omr_result_dataframe
        if run_count == 0:
            omr_result = rf
        else:
            omr_result = omr_result.append(rf)
        run_count += 1
        progress.move()
        if run_count % 5 == 0:
            progress.log(f)
        progress.log(f)
    total_time = round(time.clock()-sttime, 2)
    if run_len != 0:
        print(f'total_time={total_time}  mean_time={round(total_time / run_len, 2)}')
        if result_save:
            omr_result.to_csv(result_save_file)
    return omr_result


def omr_test_one(card_form: dict,
                 card_file='',
                 debug=True,
                 display=True,
                 result_group=True
                 ):
    # image_list = card_form['image_file_list']
    # omrfile = image_list[0] if (len(image_list[0]) > 0) & (len(file) == 0) else file
    if len(card_file) == 0:
        if len(card_form['image_file_list']) > 0:
            card_file = card_form['image_file_list'][0]
    if not os.path.isfile(card_file):
        print(f'{card_file} does not exist!')
        return
    thisform = copy.deepcopy(card_form)
    thisform['iamge_file_list'] = [card_file]
    omr = OmrModel()
    omr.set_form(thisform)
    omr.set_omr_image_filename(card_file)

    omr.sys_group_result = result_group
    omr.sys_debug = debug
    omr.sys_display = display
    omr.run()
    return omr


class OmrForm:
    """
    card_form = {
        'image_file_list': omr_image_list,
        'iamge_clip':{
            'do_clip': False,
            'x_start': 0, 'x_end': 100, 'y_start': 0, 'y_end': 200
            }
        'mark_format': {
            'mark_col_number': 37,
            'mark_row_number': 14,
            'mark_valid_area_col_start': 23,
            'mark_valid_area_col_end': 36,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 13
            },
        'group_format': {No: [(r,c),   # start position: (row number, column number)
                              int      # length
                              char     # direction, 'V'=vertical, 'H'=horizonal
                              str      # codestring,  for example: 'ABCD', '0123456789'
                              char     # choice mode, 'S'=single choice, 'M'=multi choice
                              ]}
    }
    of = OmrForm()
    of.set_file_list(image_file_list)
    of.set_mark_format(mark_format_dict)
    of.set_group_format(group_format_dict)
    form=of.get_form()
    ------
    painting format:
    # : no block painted in a group
    * : invalid painting in a group (more than one block painted for single mode 'S')
    """

    def __init__(self):
        self.form = dict()
        self.file_list = list()
        self.mark_format = dict()
        self.group_format = dict()
        self.image_clip = {
            'do_clip': False,
            'x_start': 0,
            'x_end': -1,
            'y_start': 0,
            'y_end': -1
        }

    @classmethod
    def help(cls):
        print(cls.__doc__)

    def set_file_list(self, file_name_list: list):
        self.file_list = file_name_list

    def set_image_clip(self,
                       clip_x_start,
                       clip_x_end,
                       clip_y_start,
                       clip_y_end):
        self.image_clip = {
            'do_clip': True,
            'x_start': clip_x_start,
            'x_end': clip_x_end,
            'y_start': clip_y_start,
            'y_end': clip_y_end
        }

    def set_mark_format(self,
                        mark_col_number: int,
                        mark_row_number: int,
                        mark_valid_area_col_start: int,
                        mark_valid_area_col_end: int,
                        mark_valid_area_row_start: int,
                        mark_valid_area_row_end: int

                        ):
        self.mark_format = {
            'mark_col_number': mark_col_number,
            'mark_row_number': mark_row_number,
            'mark_valid_area_col_start': mark_valid_area_col_start,
            'mark_valid_area_col_end': mark_valid_area_col_end,
            'mark_valid_area_row_start': mark_valid_area_row_start,
            'mark_valid_area_row_end': mark_valid_area_row_end
        }

    def set_group_format(self, group_dict: dict):
        self.group_format = group_dict

    def get_form(self):
        self.form = {
            'image_file_list': self.file_list,
            'image_clip': self.image_clip,
            'mark_format': self.mark_format,
            'group_format': self.group_format
        }
        return self.form


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class OmrModel(object):
    """
    processing omr image class
    set data: set_img, set_format
        imgfile: omr image file name string
        format: [mark horizon nunber, mark vertical number, valid_h_start,end, valid_v_start,end]
        savedatapath: string, path to save omr block images file in save_result_omriamge()
    set para:
        display: bool, display meassage in runtime
        logwrite: bool, write to logger, not implemented yet
    result:
        omrdict: dict, (x,y):omrblcok image matrix ndarray
        omr_recog_data: dict
        omrxypos: list, [[x-start-pos,], [x-end-pos,],[y-start-pos,], [y-end-pos,]]
        xmap: list
        ymap: list
        mark_omrimage:
        recog_omrimage:_
    inner para:
        omr_threshold:int, gray level to judge painted block
        check_
    """
    omr_code_standard_dict = \
        {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
         'F': 'BC',
         'G': 'ABC',
         'H': 'AB',
         'I': 'AD',
         'J': 'BD',
         'K': 'ABD',
         'L': 'CD',
         'M': 'ACD',
         'N': 'BCD',
         'O': 'ABCD',
         'P': 'AC',
         'Q': 'AE',
         'R': 'BE',
         'S': 'ABE',
         'T': 'CE',
         'U': 'ACE',
         'V': 'BCE',
         'W': 'ABCE',
         'X': 'DE',
         'Y': 'ADE',
         'Z': 'BDE',
         '[': 'ABDE',
         '\\': 'CDE',
         ']': 'ACDE',
         '^': 'BCDE',
         '_': 'ABCDE',
         '.': '',  # no choice
         '>': '*'  # error choice
         }

    def __init__(self):
        # input data and set parameters
        self.image_filename = ''
        self.image_rawcard = None
        self.image_card_2dmatrix = None  # np.zeros([3, 3])
        self.image_blackground_with_rawblock = None
        self.image_blackground_with_recogblock = None
        # self.image_recog_blocks = None

        # omr form parameters
        self.omr_form_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_form_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        self.omr_form_group_form_dict = {1: [(0, 0), 4, 'H', 'ABCD', 'S']}  # pos, len, dir, code, mode
        self.omr_form_group_coord_map_dict = {}
        self.omr_image_clip = False
        self.omr_image_clip_area = []

        # system control parameters
        self.sys_debug: bool = False
        self.sys_group_result = False
        self.sys_display: bool = False        # display time, error messages in running process
        self.sys_logwrite: bool = False       # record processing messages in log file, finished later

        # inner parameter
        self.check_threshold: int = 35
        self.check_vertical_window: int = 30
        self.check_horizon_window: int = 20
        self.check_step: int = 5
        self.check_moving_block_for_saturability = False

        # check position data
        self.pos_x_prj_list: list = []
        self.pos_y_prj_list: list = []
        self.pos_xy_start_end_list: list = [[], [], [], []]
        self.pos_prj_log_dict = dict()

        # recog result data
        self.omr_result_coord_blockimage_dict = {}
        self.omr_result_coord_markimage_dict = {}
        self.omr_result_data_dict = {}
        self.omr_result_dataframe = None
        self.omr_result_dataframe_content = None
        self.omr_result_save_blockimage_path = ''

        # omr encoding dict
        self.omr_code_encoding_dict = {self.omr_code_standard_dict[k]: k for k in self.omr_code_standard_dict}

    def run(self):
        # initiate some variables
        self.pos_xy_start_end_list = [[], [], [], []]
        self.omr_result_dataframe = None
        self.omr_result_dataframe_content = None

        # start running
        if self.sys_display:
            st = time.clock()
        self.get_card_image(self.image_filename)
        self.get_mark_pos()     # call check_mark_... as submethods
        self.get_coord_blockimage_dict()
        self.get_recog_data()
        self.get_result_dataframe()

        # do in plot_fun
        # self.get_recog_omrimage()
        if self.sys_display:
            print(f'running consume {time.clock()-st} seconds')

    def set_form(self, card_form):
        mark_format = [v for v in card_form['mark_format'].values()]
        group = card_form['group_format']
        self.set_mark_format(tuple(mark_format))
        self.set_group(group)
        self.omr_image_clip = card_form['image_clip']['do_clip']
        area_xend = card_form['image_clip']['x_end']
        area_yend = card_form['image_clip']['y_end']
        # area_xend = area_xend if area_xend > 0 else 100000
        # area_yend = area_xend if area_yend > 0 else 100000
        self.omr_image_clip_area = [card_form['image_clip']['x_start'],
                                    area_xend,
                                    card_form['image_clip']['y_start'],
                                    area_yend]

    def set_mark_format(self, mark_format: tuple):
        """
        :param
            card_form = [mark_h_num, mark_v_num,
                         valid_h_start, valid_h_end,
                         valid_v_start, valid_v_end]
        :return
            False and error messages if set data is error
            for example: valid position is not in mark area
        """
        if (mark_format[2] < 1) | (mark_format[3] < mark_format[2]) | (mark_format[3] > mark_format[0]):
            print(f'mark setting error: mark start{mark_format[2]}, end{mark_format[3]}')
            return
        if (mark_format[4] < 1) | (mark_format[5] < mark_format[4]) | (mark_format[5] > mark_format[1]):
            print(f'mark setting error: mark start{mark_format[2]}, end{mark_format[3]}')
            return
        self.omr_form_mark_area['mark_horizon_number'] = mark_format[0]
        self.omr_form_mark_area['mark_vertical_number'] = mark_format[1]
        self.omr_form_valid_area['mark_horizon_number'] = [mark_format[2], mark_format[3]]
        self.omr_form_valid_area['mark_vertical_number'] = [mark_format[4], mark_format[5]]

    def set_group(self, group: dict):
        """
        :param group: {g_no:[g_pos(row, col), g_len:int, g_direction:'V' or 'H', g_codestr:str,
                       g_mode:'S'/'M'], ...}
        g_no:int,  serial No for groups
        g_pos: (row, col), 1 ... maxno, start coordinate for each painting group,
        g_len: length for each group
        g_codestr: code string for painting block i.e. 'ABCD', '0123456789'
        g_direction: 'V' or 'H' for vertical or hironal direction
        g_mode: 'S' or 'M' for single or multi-choice
        """
        if type(group) != dict:
            print('error: group_format is not a dict!')
            return
        self.omr_form_group_form_dict = group
        # self.omr_code_valid_number = 0
        for gno in group.keys():
            if (type(group[gno][0]) not in [tuple, list]) | \
                    (len(group[gno][0]) != 2):
                print('error: group-pos, group_format[0] is nor tuple like (r, c)!')
                return
            if len(group[gno]) != 5:
                print('error: group_format is not tuple length=5 !')
                return
            if type(group[gno][1]) != int:
                print('error: group-len, group_format[1]\'s type is not int!')
                return
            if type(group[gno][2]) != str:
                print('error: group-code, group_format[2]\'s type is not str!')
                return
            if type(group[gno][3]) != str:
                print('error: group-mode, group_format[3]\'s type is not str!')
                return
            # if self.omr_form_group_form_dict[gno][4] == 'M':
            #    self.omr_code_valid_number = self.omr_code_valid_number + \
            #                                 group[gno][1]
            for j in range(self.omr_form_group_form_dict[gno][1]):
                # get pos coordination (row, col)
                x, y = self.omr_form_group_form_dict[gno][0]
                # add -1 to set to 0 ... n-1 mode
                x, y = (x+j-1, y-1) if self.omr_form_group_form_dict[gno][2] == 'V' else (x - 1, y + j - 1)
                # create (x, y):[gno, code, mode]
                self.omr_form_group_coord_map_dict[(x, y)] = \
                    (gno, self.omr_form_group_form_dict[gno][3][j], self.omr_form_group_form_dict[gno][4])
                # check (x, y) in mark area
                hscope = self.omr_form_valid_area['mark_horizon_number']
                vscope = self.omr_form_valid_area['mark_vertical_number']
                if (y not in range(hscope[1])) | (x not in range(vscope[1])):
                    print(f'group set error: ({x}, {y}) not in valid mark area{vscope}, {hscope}!')
            # self.omr_code_valid_number = 0
            # gno = 0
            # for k in self.omr_form_group_coord_map_dict.keys():
                # v = self.omr_form_group_coord_map_dict[k]
                # if v[2] == 'S' and v[0] != gno:
                #    self.omr_code_valid_number = self.omr_code_valid_number + 1
                # gno = v[0] if v[0] > 0 else 0

    def set_omr_image_filename(self, file_name: str):
        self.image_filename = file_name

    def get_card_image(self, image_file):
        self.image_rawcard = mg.imread(image_file)
        self.image_card_2dmatrix = self.image_rawcard
        if self.omr_image_clip:
            self.image_card_2dmatrix = self.image_rawcard[
                                       self.omr_image_clip_area[2]:self.omr_image_clip_area[3],
                                       self.omr_image_clip_area[0]:self.omr_image_clip_area[1]]
        self.image_card_2dmatrix = 255 - self.image_card_2dmatrix
        # image: 3d to 2d
        if len(self.image_card_2dmatrix.shape) == 3:
            self.image_card_2dmatrix = self.image_card_2dmatrix.mean(axis=2)

    def save_result_omriamge(self):
        if self.omr_result_save_blockimage_path == '':
            print('to set save data path!')
            return
        if not os.path.exists(self.omr_result_save_blockimage_path):
            print(f'save data path "{self.omr_result_save_blockimage_path}" not exist!')
            return
        for coord in self.omr_result_coord_blockimage_dict:
            f = self.omr_result_save_blockimage_path + '/omr_block_' + str(coord) + '_' + \
                Tools.find_file(self.image_filename)
            mg.imsave(f, self.omr_result_coord_blockimage_dict[coord])

    def get_mark_pos(self):
        # initiate omrxypos
        self.pos_xy_start_end_list = [[], [], [], []]
        # r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_seek_pos(self.image_card_2dmatrix,
                                                     rowmark=True,
                                                     step=self.check_step,
                                                     window=self.check_horizon_window)
        if _count < 0:
            return
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom zone to create map-fun for removing noise
        rownum = self.image_card_2dmatrix.shape[0]
        rownum = rownum - _step * _count
        r2, step, count = self.check_mark_seek_pos(self.image_card_2dmatrix[0:rownum, :],
                                                   rowmark=False,
                                                   step=self.check_step,
                                                   window=self.check_vertical_window)
        if count >= 0:
            if (len(r1[0]) > 0) | (len(r2[0]) > 0):
                self.pos_xy_start_end_list = np.array([r1[0], r1[1], r2[0], r2[1]])
                # adjust peak width <<unused provisionally>>
                # self.check_mark_adjustpeak()

    def check_mark_seek_pos(self, img, rowmark, step, window):
        direction = 'horizon' if rowmark else 'vertical'
        w = window
        maxlen = self.image_card_2dmatrix.shape[0] if rowmark else self.image_card_2dmatrix.shape[1]
        mark_start_end_position = [[], []]
        count = 0
        while True:
            # no mark area found
            if maxlen < w + step * count:
                if self.sys_display:
                    print(f'checking marks fail: {direction}',
                          f'imagezone= {maxlen- w - step*count}:{maxlen  - step*count}',
                          f'count={count}, step={step}, window={window}!')
                break
            imgmap = img[maxlen - w - step * count:maxlen - step * count, :].sum(axis=0) \
                if rowmark else \
                img[:, maxlen - w - step * count:maxlen - step * count].sum(axis=1)
            if self.sys_debug:
                self.pos_prj_log_dict.update({('row' if rowmark else 'col', count): imgmap.copy()})
            mark_start_end_position = self.check_mark_seek_pos_conv(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    if self.sys_display:
                        print(f'checked mark: {direction}，step={step},count={count}',
                              f'imgzone={maxlen - w - step * count}:{maxlen - step * count}',
                              f'checked_mark={len(mark_start_end_position[0])}')
                    return mark_start_end_position, step, count
            count += 1
        if self.sys_display:
            mark_number = self.omr_form_mark_area['mark_horizon_number'] \
                          if rowmark else \
                          self.omr_form_mark_area['mark_vertical_number']
            print(f'checking marks fail: found mark={len(mark_start_end_position[0])}',
                  f'defined mark={mark_number}')
        return [[], []], step, -1

    def check_mark_seek_pos_conv(self, pixel_map_vec, rowmark) -> tuple:
        img_zone_pixel_map_mean = pixel_map_vec.mean()
        pixel_map_vec[pixel_map_vec < img_zone_pixel_map_mean] = 0
        pixel_map_vec[pixel_map_vec >= img_zone_pixel_map_mean] = 1
        # smooth sharp peak and valley.
        pixel_map_vec = self.check_mark_mapfun_smoothsharp(pixel_map_vec)
        if rowmark:
            self.pos_x_prj_list = pixel_map_vec
        else:
            self.pos_y_prj_list = pixel_map_vec
        # check mark positions. with opposite direction in convolve template
        mark_start_template = np.array([1, 1, 1, -1, -1])
        mark_end_template = np.array([-1, -1, 1, 1, 1])
        judg_value = 3
        r1 = np.convolve(pixel_map_vec, mark_start_template, 'valid')
        r2 = np.convolve(pixel_map_vec, mark_end_template, 'valid')
        # mark_position = np.where(r == 3), center point is the pos
        return np.where(r1 == judg_value)[0] + 2, np.where(r2 == judg_value)[0] + 2

    def check_mark_peak_adjust(self):
        # not neccessary
        # return
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('adjust peak fail: no position vector created!')
            return
        peaknum = len(self.pos_xy_start_end_list[0])
        pw = np.array([self.pos_xy_start_end_list[1][i] - self.pos_xy_start_end_list[0][i]
                       for i in range(peaknum)])
        vw = np.array([self.pos_xy_start_end_list[0][i + 1] - self.pos_xy_start_end_list[1][i]
                       for i in range(peaknum - 1)])
        mpw = int(pw.mean())
        mvw = int(vw.mean())
        # reduce wider peak
        for i in range(peaknum - 1):
            if pw[i] > mpw + 3:
                if vw[i] < mvw:
                    self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] - (pw[i] - mpw)
                    self.pos_x_prj_list[self.pos_xy_start_end_list[1][i]:self.pos_xy_start_end_list[1][i] + mpw] = 0
                else:
                    self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] + (pw[i] - mpw)
                    self.pos_x_prj_list[self.pos_xy_start_end_list[0][i] - mpw:self.pos_xy_start_end_list[1][i]] = 0
        # move peak
        vw = [self.pos_xy_start_end_list[0][i + 1] - self.pos_xy_start_end_list[1][i] for i in range(peaknum - 1)]
        # not move first and last peak
        for i in range(1, peaknum-1):
            # move left
            if vw[i-1] > vw[i] + 3:
                self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] - 3
                self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] - 3
                self.pos_x_prj_list[self.pos_xy_start_end_list[0][i]:self.pos_xy_start_end_list[0][i] + 3] = 1
                self.pos_x_prj_list[self.pos_xy_start_end_list[1][i]:self.pos_xy_start_end_list[1][i] + 3 + 1] = 0
                if self.sys_display:
                    print(f'move peak{i} to left')
            # move right
            if vw[i] > vw[i-1] + 3:
                self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] + 3
                self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] + 3
                self.pos_x_prj_list[self.pos_xy_start_end_list[0][i] - 3:self.pos_xy_start_end_list[0][i]] = 0
                self.pos_x_prj_list[self.pos_xy_start_end_list[1][i] - 3:self.pos_xy_start_end_list[1][i]] = 1
                if self.sys_display:
                    print(f'move peak{i} to right')

    @staticmethod
    def check_mark_mapfun_smoothsharp(mapf):
        rmap = np.copy(mapf)
        # remove sharp peak -1-
        smooth_template = [-1, 2, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 0

        # remove sharp peak -11-
        smooth_template = [-1, 2, 2, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        find_pos = np.where(ck == 4)[0]
        if len(find_pos) > 0:
            rmap[find_pos[0]+1:find_pos[0]+3] = 0
        # rmap[np.where(ck == 4)[0] + 2] = 0

        # remove sharp peak -111-
        smooth_template = [-1, -1, 1, 1, 1, -1, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        find_pos = np.where(ck == 3)[0]
        if len(find_pos) > 0:
            rmap[find_pos[0]+2:find_pos[0]+4] = 0

        # remove sharp peak -1111-
        smooth_template = [-1, -1, 1, 1, 1, 1, -1, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        find_pos = np.where(ck == 4)[0]
        if len(find_pos) > 0:
            rmap[find_pos[0]+2: find_pos[0]+6] = 0

        # remove sharp peak -11111-  # 5*1
        smooth_template = [-1, -1, 1, 1, 1, 1, 1, -1, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        find_pos = np.where(ck == 5)[0]
        if len(find_pos) > 0:
            rmap[find_pos[0]+2: find_pos[0]+7] = 0

        # fill sharp valley -0-
        smooth_template = [1, -2, 1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 1
        # remove start down and end up semi-peak
        for j in range(10, 1, -1):
            if sum(rmap[:j]) == j:
                rmap[:j] = 0
                break
        for j in range(-1, -11, -1):
            if sum(rmap[j:]) == -j:
                rmap[j:] = 0
                break
        return rmap

    def check_mark_result_evaluate(self, rowmark, poslist, step, count):
        poslen = len(poslist[0])
        window = self.check_horizon_window if rowmark else self.check_vertical_window
        imgwid = self.image_card_2dmatrix.shape[0] if rowmark else self.image_card_2dmatrix.shape[1]
        hvs = 'horizon:' if rowmark else 'vertical:'
        # start position number is not same with end posistion number
        if poslen != len(poslist[1]):
            if self.sys_display:
                print(f'{hvs} start pos num({len(poslist[0])}) != end pos num({len(poslist[1])})',
                      f'step={step},count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # pos error: start pos less than end pos
        for pi in range(poslen):
            if poslist[0][pi] > poslist[1][pi]:
                if self.sys_display:
                    print(f'{hvs} start pos is less than end pos, step={step},count={count}',
                          f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
                return False
        # width > 4 is considered valid mark block.
        tl = np.array([abs(x1 - x2) for x1, x2 in zip(poslist[0], poslist[1])])
        validnum = len(tl[tl > 4])
        set_num = self.omr_form_mark_area['mark_horizon_number'] \
            if rowmark else \
            self.omr_form_mark_area['mark_vertical_number']
        if validnum != set_num:
            if self.sys_display:
                # ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} mark valid num({validnum}) != set_num({set_num})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        if len(tl) != set_num:
            if self.sys_display:
                print(f'{hvs}checked mark num({len(tl)}) != set_num({set_num})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # max width is too bigger than min width is a error result. 20%(3-5 pixels)
        maxwid = max(tl)
        minwid = min(tl)
        widratio = minwid/maxwid
        if widratio < 0.2:
            if self.sys_display:
                # ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} maxwid/minwid = {maxwid}/{minwid}',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # check max gap between 2 peaks  <<deprecated provisionally>>
        '''
        p1, p2 = self.omr_valid_area['mark_horizon_number'] \
            if rowmark else \
            self.omr_valid_area['mark_vertical_number']
        tc = np.array([poslist[0][i+1] - poslist[0][i] for i in range(p1-1, p2)])
        # tc = np.array([poslist[0][i+1] - poslist[0][i] for i in range(poslen-1)])
        maxval = max(tc)
        minval = min(tc)
        gapratio = round(maxval/minval, 2)
        # r = round(gapratio, 2)
        if gapratio > 5:
            if self.display:
                print(f'{hvs} mark gap is singular! max/min = {gapratio}',
                      f'step={step},count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
       '''
        return True

    def get_coord_blockimage_dict(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('create omrdict fail:no position vector created!')
            return
        # valid area: cut area for painting points
        for x in range(self.omr_form_valid_area['mark_horizon_number'][0]-1,
                       self.omr_form_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_form_valid_area['mark_vertical_number'][0]-1,
                           self.omr_form_valid_area['mark_vertical_number'][1]):
                self.omr_result_coord_blockimage_dict[(y, x)] = \
                    self.image_card_2dmatrix[self.pos_xy_start_end_list[2][y]:
                                             self.pos_xy_start_end_list[3][y] + 1,
                                             self.pos_xy_start_end_list[0][x]:
                                             self.pos_xy_start_end_list[1][x] + 1]
        # mark area: mark edge points
        for x in range(self.omr_form_mark_area['mark_horizon_number']):
            for y in range(self.omr_form_mark_area['mark_vertical_number']):
                self.omr_result_coord_markimage_dict[(y, x)] = \
                    self.image_card_2dmatrix[self.pos_xy_start_end_list[2][y]:
                                             self.pos_xy_start_end_list[3][y] + 1,
                                             self.pos_xy_start_end_list[0][x]:
                                             self.pos_xy_start_end_list[1][x] + 1]

    def get_block_satu2(self, bmat, row, col):
        if self.check_moving_block_for_saturability:
            return self.get_block_saturability(bmat)
        xs = self.pos_xy_start_end_list[2][row]
        xe = self.pos_xy_start_end_list[3][row] + 1
        ys = self.pos_xy_start_end_list[0][col]
        ye = self.pos_xy_start_end_list[1][col] + 1
        # origin
        sa = self.get_block_saturability(bmat)
        if sa[0] > 120:
            return sa
        # move left
        bmat = self.image_card_2dmatrix[xs:xe, ys - 2:ye - 2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move right
        bmat = self.image_card_2dmatrix[xs:xe, ys + 2:ye + 2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move up
        bmat = self.image_card_2dmatrix[xs - 2:xe - 2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move down
        bmat = self.image_card_2dmatrix[xs + 2:xe + 2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        return sa

    def get_block_saturability(self, blockmat):
        # get 0-1 image with threshold
        block01 = self.fun_normto01(blockmat, self.check_threshold)
        # feature1: mean level
        # use coefficient 10/255 normalizing
        coeff0 = 9/255
        st0 = round(blockmat.mean() * coeff0, 2)
        # feature2: big-mean-line_ratio in row or col
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        th = self.check_threshold
        # r1 = len(rowmean[rowmean > th]) / len(rowmean)
        # r2 = len(colmean[colmean > th]) / len(colmean)
        # st1 = round(max(r1, r2), 2)
        st11 = round(len(rowmean[rowmean > th]) / len(rowmean), 2)
        st12 = round(len(colmean[colmean > th]) / len(colmean), 2)
        # st1 = round(max(r1, r2), 2)
        # feature3: big-pixel-ratio
        bignum = len(blockmat[blockmat > self.check_threshold])
        st2 = round(bignum / blockmat.size, 2)
        # feature4: hole-number
        st3 = self.fun_detect_hole(block01)
        # saturational area is more than 3
        th = self.check_threshold  # 50
        # feature5: saturation area exists
        # st4 = cv2.filter2D(p, -1, np.ones([3, 5]))
        st4 = filters.convolve(self.fun_normto01(blockmat, th),
                               np.ones([3, 5]), mode='constant')
        st4 = 1 if len(st4[st4 >= 14]) >= 1 else 0
        return st0, st11, st12, st2, st3, st4

    @staticmethod
    def fun_detect_hole(mat):
        # 3x4 hole
        m = np.array([[1,  1,  1,  1],
                      [1, -1, -1,  1],
                      [1,  1,  1,  1]])
        rf = filters.convolve(mat, m, mode='constant')
        r0 = len(rf[rf == 10])
        # r0 = 1 if len(rf[rf == 10]) > 0 else 0
        # 3x5 hole
        m = np.array([[1,  1,  1,  1, 1],
                     [1, -1, -1, -1, 1],
                     [1,  1,  1,  1, 1]])
        rf = filters.convolve(mat, m, mode='constant')
        r1 = len(rf[rf == 12])
        if r1 == 0:
            m = np.array([[1,  1,  1, 1, 1],
                         [1, -1, -1, 0, 1],
                         [1,  1,  1, 1, 1]])
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        if r1 == 0:
            m = np.array([[1,  1,  1,  1, 1],
                         [1,  0, -1, -1, 1],
                         [1,  1,  1,  1, 1]])
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        # 4x5 hole
        m = np.array([[0,  1,  1,  1, 0],
                     [1,  0, -1, -1, 1],
                     [1,  0, -1, -1, 1],
                     [0,  1,  1,  1, 0]])
        rf = filters.convolve(mat, m, mode='constant')
        r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1, 1, 0],
                         [1, -1, -1, 0, 1],
                         [1, -1, -1, 0, 1],
                         [0,  1,  1, 1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1, 1, 0],
                         [1,  0,  0, 0, 1],
                         [1, -1, -1, -1, 1],
                         [0,  1,  1, 1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1,  1, 0],
                         [1, -1, -1, -1, 1],
                         [1,  0,  0,  0, 1],
                         [0,  1,  1,  1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        return r0 + (1 if r1 > 0 else 0) + (1 if r2 > 0 else 0)

    @staticmethod
    def fun_normto01(mat, th):
        m = np.copy(mat)
        m[m < th] = 0
        m[m >= th] = 1
        return m

    def get_image_with_rawblocks(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('no position vector created! so cannot create omrdict!')
            return
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.image_card_2dmatrix.shape)
        '''
        for col in range(self.omr_form_valid_area['mark_horizon_number'][0]-1,
                         self.omr_form_valid_area['mark_horizon_number'][1]):
            for row in range(self.omr_form_valid_area['mark_vertical_number'][0]-1,
                             self.omr_form_valid_area['mark_vertical_number'][1]):
                omrimage[self.pos_xy_start_end_list[2][row]: self.pos_xy_start_end_list[3][row] + 1,
                         self.pos_xy_start_end_list[0][col]: self.pos_xy_start_end_list[1][col] + 1] = \
                         self.omr_result_coord_blockimage_dict[(row, col)]
        '''
        for col in range(self.omr_form_mark_area['mark_horizon_number']):
            for row in range(self.omr_form_mark_area['mark_vertical_number']):
                omrimage[self.pos_xy_start_end_list[2][row]: self.pos_xy_start_end_list[3][row] + 1,
                         self.pos_xy_start_end_list[0][col]: self.pos_xy_start_end_list[1][col] + 1] = \
                    self.image_card_2dmatrix[self.pos_xy_start_end_list[2][row]:
                                             self.pos_xy_start_end_list[3][row] + 1,
                                             self.pos_xy_start_end_list[0][col]:
                                             self.pos_xy_start_end_list[1][col] + 1]
        # self.image_recog_blocks = omrimage
        self.image_blackground_with_rawblock = omrimage
        return omrimage

    def get_image_with_recogblocks(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('no position vector created! so cannot create recog_omr_image!')
            return
        # init image with zero background
        omr_recog_block = np.zeros(self.image_card_2dmatrix.shape)
        # set block_area with painted block in raw image
        marknum = len(self.omr_result_data_dict['label'])
        for p in range(marknum):
            if self.omr_result_data_dict['label'][p] == 1:
                _x, _y = self.omr_result_data_dict['coord'][p]
                h1, h2 = [self.omr_form_valid_area['mark_horizon_number'][0] - 1,
                          self.omr_form_valid_area['mark_horizon_number'][1]]
                v1, v2 = [self.omr_form_valid_area['mark_vertical_number'][0] - 1,
                          self.omr_form_valid_area['mark_vertical_number'][1]]
                if (_x in range(v1, v2)) & (_y in range(h1, h2)):
                    omr_recog_block[self.pos_xy_start_end_list[2][_x]:
                                    self.pos_xy_start_end_list[3][_x] + 1,
                                    self.pos_xy_start_end_list[0][_y]:
                                    self.pos_xy_start_end_list[1][_y] + 1] \
                        = self.omr_result_coord_blockimage_dict[(_x, _y)]
            p += 1
        self.image_blackground_with_recogblock = omr_recog_block
        return omr_recog_block

    # create recog_data, and test use svm in sklearn
    def get_recog_data(self):
        self.omr_result_data_dict = {'coord': [], 'feature': [], 'group': [],
                                     'code': [], 'mode': [], 'label': []}
        lencheck = len(self.pos_xy_start_end_list[0]) * \
            len(self.pos_xy_start_end_list[1]) * \
            len(self.pos_xy_start_end_list[3]) * \
            len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('create recog_data fail: no position vector!')
            return
        # total_mean = 0
        # pnum = 0
        for j in range(self.omr_form_valid_area['mark_horizon_number'][0]-1,
                       self.omr_form_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_form_valid_area['mark_vertical_number'][0]-1,
                           self.omr_form_valid_area['mark_vertical_number'][1]):
                self.omr_result_data_dict['coord'].append((i, j))
                # whether using moving block by satu2
                self.omr_result_data_dict['feature'].append(
                    self.get_block_saturability(self.omr_result_coord_blockimage_dict[(i, j)]))
                # self.omr_recog_data['feature'].append(
                #    self.get_block_satu2(self.omrdict[(i, j)], i, j))
                if (i, j) in self.omr_form_group_coord_map_dict:
                    self.omr_result_data_dict['group'].append(self.omr_form_group_coord_map_dict[(i, j)][0])
                    self.omr_result_data_dict['code'].append(self.omr_form_group_coord_map_dict[(i, j)][1])
                    self.omr_result_data_dict['mode'].append(self.omr_form_group_coord_map_dict[(i, j)][2])
                else:
                    self.omr_result_data_dict['group'].append(-1)
                    self.omr_result_data_dict['code'].append('.')
                    self.omr_result_data_dict['mode'].append('.')
                '''
                self.omr_recog_data['group'].append((
                    self.omr_group_map[(i, j)][0] if (i, j) in self.omr_group_map else -1))
                self.omr_recog_data['code'].append((
                    self.omr_group_map[(i, j)][1] if (i, j) in self.omr_group_map else '.'))
                self.omr_recog_data['mode'].append((
                    self.omr_group_map[(i, j)][2] if (i, j) in self.omr_group_map else '.'))
               '''
        clu = KMeans(2)
        clu.fit(self.omr_result_data_dict['feature'])
        self.omr_result_data_dict['label'] = clu.predict(self.omr_result_data_dict['feature'])

    # result dataframe
    def get_result_dataframe(self):
        # set f, initiatial dataframe
        f = Tools.find_file(self.image_filename)
        if self.sys_group_result:
            self.omr_result_dataframe = \
                pd.DataFrame({'card': [f],
                              'result': ['XXX'],
                              'len': [-1],
                              'group': [-1],
                              'valid': [0]
                              })
        else:
            self.omr_result_dataframe = \
                pd.DataFrame({'card': [f],
                              'result': ['XXX'],
                              'len': [-1]
                              })

        # recog_data is error, return len=-1, code='XXX'
        if len(self.omr_result_data_dict['label']) == 0:
            if self.sys_display:
                print('result fail: recog data is not created!')
            return self.omr_result_dataframe

        # recog_data is ok, return result dataframe
        rdf = pd.DataFrame({'coord': self.omr_result_data_dict['coord'],
                            'label': self.omr_result_data_dict['label'],
                            'feat': self.omr_result_data_dict['feature'],
                            'group': self.omr_result_data_dict['group'],
                            'code': self.omr_result_data_dict['code'],
                            'mode': self.omr_result_data_dict['mode']
                            })

        # check label sign for feature
        if rdf.sort_values('feat', ascending=False).head(1)['label'].values[0] == 0:
            rdf['label'] = rdf['label'].apply(lambda x: 1 - x)

        # reverse label if all label ==1 (all blockes painted!)
        if rdf[rdf.group > 0].label.sum() == rdf[rdf.group > 0].count()[0]:
            rdf.label = 0

        # set label 0 (no painted) block to ''
        rdf.loc[rdf.label == 0, 'code'] = ''

        # create result dataframe
        outdf = rdf[rdf.group > 0].sort_values('group')[['group', 'code']].groupby('group').sum()
        rs_codelen = 0
        rs_code = []
        group_str = ''
        if len(outdf) > 0:
            out_se = outdf['code'].apply(lambda s: ''.join(sorted(list(s))))
            group_list = sorted(self.omr_form_group_form_dict.keys())
            for g in group_list:
                # ts = outdf.loc[g, 'code'].replace('.', '')
                # ts = ''.join(sorted(list(ts)))
                if g in out_se.index:
                    ts = out_se[g]
                    if len(ts) > 0:
                        rs_codelen = rs_codelen + 1
                        if len(ts) > 1:
                            if self.omr_form_group_form_dict[g][4] == 'M':
                                ts = self.omr_code_encoding_dict[ts]
                            elif self.sys_debug:  # error choice= <raw string> if debug
                                group_str = group_str + str(g) + ':[' + ts + ']_'
                                ts = self.omr_code_encoding_dict[ts]
                            else:  # error choice= '>'
                                group_str = group_str + str(g) + ':[' + ts + ']_'
                                ts = '>'
                    else:  # len(ts)==0
                        ts = '.'
                    rs_code.append(ts)
                else:
                    # group g not found
                    rs_code.append('@')
                    group_str = group_str + str(g) + ':@_'
                    # invalid = 0
            rs_code = ''.join(rs_code)
            group_str = group_str[:-1]
            invalid = 1
        # no group found
        else:
            return self.omr_result_dataframe

        # group result to dataframe: fname, len, group_str, result
        if self.sys_group_result:
            self.omr_result_dataframe = \
                pd.DataFrame({'card': [f],
                              'result': [rs_code],
                              'len': [rs_codelen],
                              'group': [group_str],
                              'valid': [invalid]
                              })
        # concise result to dataframe
        else:
            self.omr_result_dataframe = \
                pd.DataFrame({'card': [f],
                              'result': [rs_code],
                              'len': [rs_codelen]
                              })
        # debug result to debug_dataframe: fname, coordination, group, label, feature
        if self.sys_debug:
            self.omr_result_dataframe_content = rdf
        # return result
        return self.omr_result_dataframe

    # --- show omrimage or plot result data ---
    def plot_result(self):
        from pylab import subplot  # , scatter, gca, show
        # from matplotlib.ticker import MultipleLocator  # , FormatStrFormatter

        plt.figure('Omr Model:'+self.image_filename)
        # plt.title(self.image_filename)
        ax = subplot(231)
        '''
        xy_major_locator = MultipleLocator(5)  # 主刻度标签设置为5的倍数
        xy_minor_locator = MultipleLocator(1)  # 副刻度标签设置为1的倍数
        ax.xaxis.set_major_locator(xy_major_locator)
        ax.xaxis.set_major_locator(xy_minor_locator)
        ax.yaxis.set_major_locator(xy_major_locator)
        ax.yaxis.set_major_locator(xy_minor_locator)
        '''
        self.plot_image_raw_card()

        ax = subplot(232)
        self.plot_image_formed_card()
        ax = subplot(233)
        self.plot_image_recogblocks()
        ax = subplot(223)
        self.plot_mapfun_horizon_mark()
        ax = subplot(224)
        self.plot_mapfun_vertical_mark()
        # ax = subplot(338)
        # self.plot_grid_blockpoints()

    def plot_image_raw_card(self):
        # plt.figure(0)
        # plt.title(self.image_filename)
        if type(self.image_rawcard) != np.ndarray:
            print('no raw card image file')
            return
        plt.imshow(self.image_rawcard)

    def plot_image_formed_card(self):
        # plt.figure(1)
        # plt.title(self.image_filename)
        plt.imshow(self.image_card_2dmatrix)

    def plot_mapfun_horizon_mark(self):
        # plt.figure(2)
        plt.xlabel('horizon mark map fun')
        plt.plot(self.pos_x_prj_list)

    def plot_mapfun_vertical_mark(self):
        # plt.figure(3)
        plt.xlabel('vertical mark map fun')
        plt.plot(self.pos_y_prj_list)

    def plot_image_rawblocks(self):
        # if type(self.mark_omr_image) != np.ndarray:
        #    self.get_image_blackground_blockimage()
        # plt.figure(4)
        plt.title('recognized - omr - region ' + self.image_filename)
        # plt.imshow(self.mark_omr_image)
        plt.imshow(self.get_image_with_rawblocks())

    def plot_image_recogblocks(self):
        if type(self.image_blackground_with_recogblock) != np.ndarray:
            self.get_image_with_recogblocks()
        # plt.figure('recog block image')
        # plt.title('recognized - omr - region' + self.image_filename)
        plt.imshow(self.image_blackground_with_recogblock)

    def plot_image_markline_recogblock(self):
        # plt.figure('markline')
        plt.title(self.image_filename)
        plt.imshow(self.image_card_2dmatrix)
        xset = np.concatenate([self.pos_xy_start_end_list[0], self.pos_xy_start_end_list[1]])
        yset = np.concatenate([self.pos_xy_start_end_list[2], self.pos_xy_start_end_list[3]])
        xrange = [x for x in range(self.image_card_2dmatrix.shape[1])]
        yrange = [y for y in range(self.image_card_2dmatrix.shape[0])]
        for x in xset:
            plt.plot([x]*len(yrange), yrange)
        for y in yset:
            plt.plot(xrange, [y]*len(xrange))

    def plot_grid_blockpoints(self):
        from pylab import subplot, scatter, gca, show
        from matplotlib.ticker import MultipleLocator  # , FormatStrFormatter
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        # plt.figure('markgrid')
        plt.title(self.image_filename)
        data_mean = np.array(self.omr_result_data_dict['feature'])[:, 0]
        data_coord = np.array(self.omr_result_data_dict['coord'])
        x, y, z = [], [], []
        for i, lab in enumerate(self.omr_result_data_dict['label']):
            if lab == 1:
                x.append(data_coord[i, 0])
                y.append(data_coord[i, 1])
                z.append(data_mean[i])
        xy_major_locator = MultipleLocator(5)  # 主刻度标签设置为5的倍数
        xy_minor_locator = MultipleLocator(1)  # 副刻度标签设置为1的倍数

        ax = subplot(111)
        ax.xaxis.set_major_locator(xy_major_locator)
        ax.xaxis.set_major_locator(xy_minor_locator)
        ax.yaxis.set_major_locator(xy_major_locator)
        ax.yaxis.set_major_locator(xy_minor_locator)

        scatter(y, x)  # , c=z, cmap=cm)
        gca().invert_yaxis()

        # gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1d'))
        ax.xaxis.grid(b=True, which='minor', color='red', linestyle='dashed')   # x坐标轴的网格使用主副刻度
        ax.yaxis.grid(b=True, which='major')    # y坐标轴的网格使用主刻度
        ax.grid(color='gray', linestyle='-', linewidth=1)
        show()


class Tools:
    # --- some useful functions in omrmodel or outside
    @staticmethod
    def show_image(fstr):
        if os.path.isfile(fstr):
            plt.imshow(mg.imread(fstr))
            plt.title(fstr)
            plt.show()
        else:
            print(f'file \"{fstr}\" is not found!')

    @staticmethod
    def find_file(path_file):
        return path_file.replace('/', '\\').split('\\')[-1]

    @staticmethod
    def find_path(path_file):
        ts = Tools.find_file(path_file)
        return path_file.replace(ts, '')
    # class Tools end

    @staticmethod
    def matrix_row_reverse(matrix_2d):
        return matrix_2d[::-1]

    @staticmethod
    def matrix_col_reverse(matrix_2d):
        return np.array([matrix_2d[r, :][::-1] for r in range(matrix_2d.shape[0])])

    @staticmethod
    def matrix_rotate90_right(matrix_2d):
        # matrix_2d[:] = np.array(map(list, zip(*matrix_2d[::-1])))
        temp = map(list, zip(*matrix_2d[::-1]))
        return np.array(list(temp))


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s=''):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        if len(s) > 0:
            print(s)
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('>' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()
