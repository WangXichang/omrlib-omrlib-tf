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
# import gc
# import tensorflow as tf
# import cv2
# from PIL import Image
# from sklearn import svm


def help_OmrModel():
    print(OmrModel.__doc__)


def help_OmrForm():
    print(OmrForm.__doc__)

def help_read_batch():
    print(omr_read_batch.__doc__)


def help_omr_table():
    return OmrModel.omr_map


def omr_read_batch(card_form: dict,
                   result_group=False,
                   result_save=False,
                   result_save_file='omr_data'):
    """
    :input
        card_form: form_dict, could get from class OmrForm
        result_group: bool, False=no group info in result_dataframe
        result_save: bool, True=save result dataframe to disk file(result_save_file)
        result_save_file: file name to save data, auto added .csv
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
    # set model
    omr = OmrModel()
    omr.set_form(card_form)
    omr.group_result = result_group
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
        omr.set_img(f)
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


def omr_read_one(card_form: dict,
                 file='',
                 result_group=False,
                 debug=False):
    image_list = card_form['image_file_list']
    omrfile = image_list[0] if (len(image_list[0]) > 0) & (len(file) == 0) else file
    if not os.path.isfile(omrfile):
        print(f'{omrfile} does not exist!')
        return
    # mark_format = [v for v in card_form['mark_format'].values()]
    # group = card_form['group_format']
    omr = OmrModel()
    omr.set_form(card_form)
    # omr.set_format(tuple(mark_format))
    # omr.set_group(group)

    omr.group_result = result_group
    omr.debug = debug
    omr.set_img(omrfile)
    omr.run()
    return omr.omr_result_dataframe


def omr_test_one(card_form: dict,
                 omrfile='',
                 result_group=False,
                 debug=False,
                 display=True):
    # image_list = card_form['image_file_list']
    # omrfile = image_list[0] if (len(image_list[0]) > 0) & (len(file) == 0) else file
    if not os.path.isfile(omrfile):
        print(f'{omrfile} does not exist!')
        return
    thisform = card_form.copy()
    thisform['iamge_file_list'] = [omrfile]
    omr = OmrModel()
    omr.set_form(thisform)
    omr.set_img(omrfile)

    omr.group_result = result_group
    omr.debug = debug
    omr.display = display
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
    omr_map = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
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
               '.': '',
               '>': '*'
               }

    def __init__(self):
        # input data and set parameters
        self.image_filename = ''
        self.image_2d_matrix = np.zeros([3, 3])
        self.save_data_path = ''
        # omr parameters
        self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        self.omr_group = {1: [(0, 0), 4, 'H', 'ABCD', 'S']}  # pos, len, dir, code, mode
        self.omr_group_map = {}
        self.omr_code_valid_number = 0
        self.group_str = ''.join([str(g) + ';' for g in self.omr_group.keys()])[:-1]
        # system control parameters
        self.debug: bool = False
        self.group_result = False
        self.display: bool = False        # display time, error messages in running process
        self.logwrite: bool = False       # record processing messages in log file, finished later
        # inner parameter
        self.omr_threshold: int = 35
        self.check_vertical_window: int = 30
        self.check_horizon_window: int = 20
        self.check_step: int = 5
        self.moving_block_for_saturability = False
        # result data
        self.xmap: list = []
        self.ymap: list = []
        self.omrxypos: list = [[], [], [], []]
        self.mark_omriamge = None
        self.recog_omriamge = None
        self.omrdict = {}
        self.omr_recog_data = {}
        self.omr_result_dataframe = None
        self.omr_debug_dataframe = None
        # self.omrsvm = None
        self.omr_image_clip = False
        self.omr_image_clip_area = []
        # reverse omr map
        self.omr_map_dict = {self.omr_map[k]:k for k in self.omr_map}

    def run(self):
        # initiate some variables
        self.omrxypos = [[], [], [], []]
        # start running
        st = time.clock()
        self.get_img(self.image_filename)
        self.get_markblock()
        self.get_omrdict_xyimage()
        self.get_recog_data()
        self.get_result_dataframe2()
        # self.get_mark_omrimage()
        # self.get_recog_omrimage()
        if self.display:
            print(f'consume {time.clock()-st}')

    def set_form(self, card_form):
        mark_format = [v for v in card_form['mark_format'].values()]
        group = card_form['group_format']
        self.set_format(tuple(mark_format))
        self.set_group(group)
        # self.group_str = ''.join([str(i) + ';' for i in range(len(self.omr_group.keys()))])[:-1]
        self.omr_image_clip = card_form['image_clip']['do_clip']
        area_xend = card_form['image_clip']['x_end']
        area_yend = card_form['image_clip']['y_end']
        area_xend = area_xend if area_xend > 0 else 100000
        area_yend = area_xend if area_yend > 0 else 100000
        self.omr_image_clip_area = [card_form['image_clip']['x_start'],
                                    area_xend,
                                    card_form['image_clip']['y_start'],
                                    area_yend]

    def set_format(self, mark_format: tuple):
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
        self.omr_mark_area['mark_horizon_number'] = mark_format[0]
        self.omr_mark_area['mark_vertical_number'] = mark_format[1]
        self.omr_valid_area['mark_horizon_number'] = [mark_format[2], mark_format[3]]
        self.omr_valid_area['mark_vertical_number'] = [mark_format[4], mark_format[5]]

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
        self.omr_group = group
        self.omr_code_valid_number = 0
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
            if self.omr_group[gno][4] == 'M':
                self.omr_code_valid_number = self.omr_code_valid_number + \
                                             group[gno][1]
            for j in range(self.omr_group[gno][1]):
                # get pos coordination (row, col)
                x, y = self.omr_group[gno][0]
                # add -1 to set to 0 ... n-1 mode
                x, y = (x+j-1, y-1) if self.omr_group[gno][2] == 'V' else (x-1, y+j-1)
                # create (x, y):[gno, code, mode]
                self.omr_group_map[(x, y)] = (gno, self.omr_group[gno][3][j], self.omr_group[gno][4])
                # check (x, y) in mark area
                hscope = self.omr_valid_area['mark_horizon_number']
                vscope = self.omr_valid_area['mark_vertical_number']
                if (y not in range(hscope[1])) | (x not in range(vscope[1])):
                    print(f'group set error: ({x}, {y}) not in valid mark area{vscope}, {hscope}!')
            self.omr_code_valid_number = 0
            gno = 0
            for k in self.omr_group_map.keys():
                v = self.omr_group_map[k]
                if v[2] == 'S' and v[0] != gno:
                    self.omr_code_valid_number = self.omr_code_valid_number + 1
                gno = v[0] if v[0] > 0 else 0
        self.group_str = ''.join([str(g) + ';' for g in self.omr_group.keys()])[:-1]

    def set_img(self, file_name: str):
        self.image_filename = file_name

    def get_img(self, image_file):
        self.image_2d_matrix = mg.imread(image_file)
        if self.omr_image_clip:
            self.image_2d_matrix = self.image_2d_matrix[
                                   self.omr_image_clip_area[2]:self.omr_image_clip_area[3],
                                   self.omr_image_clip_area[0]:self.omr_image_clip_area[1]]
        self.image_2d_matrix = 255 - self.image_2d_matrix  # type: np.array
        # self.img = np.array(self.img)
        if len(self.image_2d_matrix.shape) == 3:
            self.image_2d_matrix = self.image_2d_matrix.mean(axis=2)

    def save_result_omriamge(self):
        if self.save_data_path == '':
            print('to set save data path!')
            return
        if not os.path.exists(self.save_data_path):
            print(f'save data path "{self.save_data_path}" not exist!')
            return
        for coord in self.omrdict:
            f = self.save_data_path + '/omr_block_' + str(coord) + '_' + \
                Tools.find_file(self.image_filename)
            mg.imsave(f, self.omrdict[coord])

    def get_markblock(self):
        # initiate omrxypos
        self.omrxypos = [[], [], [], []]
        # r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_pos(self.image_2d_matrix,
                                                rowmark=True,
                                                step=self.check_step,
                                                window=self.check_horizon_window)
        if _count < 0:
            return
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom zone to create map-fun for removing noise
        rownum = self.image_2d_matrix.shape[0]
        rownum = rownum - _step * _count
        r2, step, count = self.check_mark_pos(self.image_2d_matrix[0:rownum, :],
                                              rowmark=False,
                                              step=self.check_step,
                                              window=self.check_vertical_window)
        if count >= 0:
            if (len(r1[0]) > 0) | (len(r2[0]) > 0):
                self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])
                # adjust peak width
                # self.check_mark_adjustpeak()

    def check_mark_pos(self, img, rowmark, step, window):
        direction = 'horizon' if rowmark else 'vertical'
        w = window
        maxlen = self.image_2d_matrix.shape[0] if rowmark else self.image_2d_matrix.shape[1]
        mark_start_end_position = [[], []]
        count = 0
        while True:
            # no mark area found
            if maxlen < w + step * count:
                if self.display:
                    print(f'checking marks fail: {direction}',
                          f'imagezone= {maxlen- w - step*count}:{maxlen  - step*count}',
                          f'count={count}, step={step}, window={window}!')
                break
            imgmap = img[maxlen - w - step * count:maxlen - step * count, :].sum(axis=0) \
                if rowmark else \
                img[:, maxlen - w - step * count:maxlen - step * count].sum(axis=1)
            mark_start_end_position = self.check_mark_block(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    if self.display:
                        print(f'checked mark: {direction}，step={step},count={count}',
                              f'imgzone={maxlen - w - step * count}:{maxlen - step * count}',
                              f'checked_mark={len(mark_start_end_position[0])}')
                    return mark_start_end_position, step, count
            count += 1
        if self.display:
            mark_number = self.omr_mark_area['mark_horizon_number'] \
                          if rowmark else \
                          self.omr_mark_area['mark_vertical_number']
            print(f'checking marks fail: found mark={len(mark_start_end_position[0])}',
                  f'defined mark={mark_number}')
        return [[], []], step, -1

    def check_mark_block(self, mapvec, rowmark) -> tuple:
        imgmapmean = mapvec.mean()
        mapvec[mapvec < imgmapmean] = 0
        mapvec[mapvec >= imgmapmean] = 1
        # smooth sharp peak and valley
        mapvec = self.check_mark_mapfun_smoothsharp(mapvec)
        if rowmark:
            self.xmap = mapvec
        else:
            self.ymap = mapvec
        # check mark positions
        mark_start_template = np.array([1, 1, 1, -1, -1])
        mark_end_template = np.array([-1, -1, 1, 1, 1])
        judg_value = 3
        r1 = np.convolve(mapvec, mark_start_template, 'valid')
        r2 = np.convolve(mapvec, mark_end_template, 'valid')
        # mark_position = np.where(r == 3)
        result = np.where(r1 == judg_value)[0] + 2, np.where(r2 == judg_value)[0] + 2
        return result

    def check_mark_adjustpeak(self):
        # not neccessary
        # return
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                         (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('adjust peak fail: no position vector created!')
            return
        peaknum = len(self.omrxypos[0])
        pw = np.array([self.omrxypos[1][i] - self.omrxypos[0][i] for i in range(peaknum)])
        vw = np.array([self.omrxypos[0][i+1] - self.omrxypos[1][i] for i in range(peaknum-1)])
        mpw = int(pw.mean())
        mvw = int(vw.mean())
        # reduce wider peak
        for i in range(peaknum - 1):
            if pw[i] > mpw + 3:
                if vw[i] < mvw:
                    self.omrxypos[1][i] = self.omrxypos[1][i] - (pw[i] - mpw)
                    self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i] + mpw] = 0
                else:
                    self.omrxypos[0][i] = self.omrxypos[0][i] + (pw[i] - mpw)
                    self.xmap[self.omrxypos[0][i] - mpw:self.omrxypos[1][i]] = 0
        # move peak
        vw = [self.omrxypos[0][i+1] - self.omrxypos[1][i] for i in range(peaknum-1)]
        # not move first and last peak
        for i in range(1, peaknum-1):
            # move left
            if vw[i-1] > vw[i] + 3:
                self.omrxypos[0][i] = self.omrxypos[0][i] - 3
                self.omrxypos[1][i] = self.omrxypos[1][i] - 3
                self.xmap[self.omrxypos[0][i]:self.omrxypos[0][i]+3] = 1
                self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i]+3+1] = 0
                if self.display:
                    print(f'move peak{i} to left')
            # move right
            if vw[i] > vw[i-1] + 3:
                self.omrxypos[0][i] = self.omrxypos[0][i] + 3
                self.omrxypos[1][i] = self.omrxypos[1][i] + 3
                self.xmap[self.omrxypos[0][i]-3:self.omrxypos[0][i]] = 0
                self.xmap[self.omrxypos[1][i]-3:self.omrxypos[1][i]] = 1
                if self.display:
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
        rmap[np.where(ck == 4)[0] + 1] = 0
        rmap[np.where(ck == 4)[0] + 2] = 0
        # remove sharp peak -111-
        smooth_template = [-1, -1, 1, 1, 1, -1, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 3)[0] + 1] = 0
        rmap[np.where(ck == 3)[0] + 2] = 0
        rmap[np.where(ck == 3)[0] + 3] = 0
        # fill sharp valley -0-
        smooth_template = [1, -2, 1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 1
        # remove sharp peak -1-
        smooth_template = [-1, 2, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 0
        return rmap

    def check_mark_result_evaluate(self, rowmark, poslist, step, count):
        poslen = len(poslist[0])
        window = self.check_horizon_window if rowmark else self.check_vertical_window
        imgwid = self.image_2d_matrix.shape[0] if rowmark else self.image_2d_matrix.shape[1]
        hvs = 'horizon:' if rowmark else 'vertical:'
        # start position number is not same with end posistion number
        if poslen != len(poslist[1]):
            if self.display:
                print(f'{hvs} start pos num({len(poslist[0])}) != end pos num({len(poslist[1])})',
                      f'step={step},count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # pos error: start pos less than end pos
        for pi in range(poslen):
            if poslist[0][pi] > poslist[1][pi]:
                if self.display:
                    print(f'{hvs} start pos is less than end pos, step={step},count={count}',
                          f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
                return False
        # width > 4 is considered valid mark block.
        tl = np.array([abs(x1 - x2) for x1, x2 in zip(poslist[0], poslist[1])])
        validnum = len(tl[tl > 4])
        setnum = self.omr_mark_area['mark_horizon_number'] \
            if rowmark else \
            self.omr_mark_area['mark_vertical_number']
        if validnum != setnum:
            if self.display:
                # ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} mark valid num({validnum}) != set_num({setnum})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        if len(tl) != setnum:
            if self.display:
                print(f'{hvs}checked mark num({len(tl)}) != set_num({setnum})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # max width is too bigger than min width is a error result. 20%(3-5 pixels)
        maxwid = max(tl)
        minwid = min(tl)
        widratio = minwid/maxwid
        if widratio < 0.2:
            if self.display:
                # ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} maxwid/minwid = {maxwid}/{minwid}',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # check max gap between 2 peaks
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
        return True

    def get_omrdict_xyimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                         (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('create omrdict fail:no position vector created!')
            return
        # cut area for painting points
        for x in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                self.omrdict[(y, x)] = \
                    self.image_2d_matrix[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                         self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_block_satu2(self, bmat, row, col):
        if self.moving_block_for_saturability:
            return self.get_block_saturability(bmat)
        xs = self.omrxypos[2][row]
        xe = self.omrxypos[3][row]+1
        ys = self.omrxypos[0][col]
        ye = self.omrxypos[1][col]+1
        # origin
        sa = self.get_block_saturability(bmat)
        if sa[0] > 120:
            return sa
        # move left
        bmat = self.image_2d_matrix[xs:xe, ys - 2:ye - 2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move right
        bmat = self.image_2d_matrix[xs:xe, ys + 2:ye + 2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move up
        bmat = self.image_2d_matrix[xs - 2:xe - 2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move down
        bmat = self.image_2d_matrix[xs + 2:xe + 2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        return sa

    def get_block_saturability(self, blockmat):
        # get 0-1 image with threshold
        block01 = self.fun_normto01(blockmat, self.omr_threshold)
        # feature1: mean level
        # use coefficient 10/255 normalizing
        coeff0 = 9/255
        st0 = round(blockmat.mean() * coeff0, 2)
        # feature2: big-mean-line_ratio in row or col
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        th = self.omr_threshold
        # r1 = len(rowmean[rowmean > th]) / len(rowmean)
        # r2 = len(colmean[colmean > th]) / len(colmean)
        # st1 = round(max(r1, r2), 2)
        st11 = round(len(rowmean[rowmean > th]) / len(rowmean), 2)
        st12 = round(len(colmean[colmean > th]) / len(colmean), 2)
        # st1 = round(max(r1, r2), 2)
        # feature3: big-pixel-ratio
        bignum = len(blockmat[blockmat > self.omr_threshold])
        st2 = round(bignum / blockmat.size, 2)
        # feature4: hole-number
        st3 = self.fun_detect_hole(block01)
        # saturational area is more than 3
        th = self.omr_threshold  # 50
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

    def get_mark_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                         (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('no position vector created! so cannot create omrdict!')
            return
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.image_2d_matrix.shape)
        for col in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                         self.omr_valid_area['mark_horizon_number'][1]):
            for row in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                             self.omr_valid_area['mark_vertical_number'][1]):
                omrimage[self.omrxypos[2][row]: self.omrxypos[3][row]+1,
                         self.omrxypos[0][col]: self.omrxypos[1][col]+1] = self.omrdict[(row, col)]
        self.mark_omriamge = omrimage

    def get_recog_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                         (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('no position vector created! so cannot create recog_omr_image!')
            return
        recogomr = np.zeros(self.image_2d_matrix.shape)
        marknum = len(self.omr_recog_data['label'])
        for p in range(marknum):
            if self.omr_recog_data['label'][p] == 1:
                _x, _y = self.omr_recog_data['coord'][p]
                h1, h2 = [self.omr_valid_area['mark_horizon_number'][0] - 1,
                          self.omr_valid_area['mark_horizon_number'][1]]
                v1, v2 = [self.omr_valid_area['mark_vertical_number'][0] - 1,
                          self.omr_valid_area['mark_vertical_number'][1]]
                if (_x in range(v1, v2)) & (_y in range(h1, h2)):
                    recogomr[self.omrxypos[2][_x]: self.omrxypos[3][_x]+1,
                             self.omrxypos[0][_y]: self.omrxypos[1][_y]+1] \
                        = self.omrdict[(_x, _y)]
            p += 1
        self.recog_omriamge = recogomr

    # create recog_data, and test use svm in sklearn
    def get_recog_data(self):
        # self.omr_recog_data = {'coord': [], 'label': [], 'feature': [],
        #                        'code': [], 'mode':[], 'group':[]}
        self.omr_recog_data = {'coord': [], 'feature': [], 'group': [],
                               'code': [], 'mode': [], 'label': []}
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
            len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                         (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('create recog_data fail: no position vector!')
            return
        # total_mean = 0
        # pnum = 0
        for j in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                # painted_mean = self.omrdict[(i, j)].mean()
                # self.omr_recog_data['bmean'].append(round(painted_mean, 2))
                self.omr_recog_data['coord'].append((i, j))
                # whether using moving block by satu2
                self.omr_recog_data['feature'].append(
                    self.get_block_saturability(self.omrdict[(i, j)]))
                # self.omr_recog_data['feature'].append(
                #    self.get_block_satu2(self.omrdict[(i, j)], i, j))
                self.omr_recog_data['group'].append((
                    self.omr_group_map[(i, j)][0] if (i, j) in self.omr_group_map else -1))
                self.omr_recog_data['code'].append((
                    self.omr_group_map[(i, j)][1] if (i, j) in self.omr_group_map else '.'))
                self.omr_recog_data['mode'].append((
                    self.omr_group_map[(i, j)][2] if (i, j) in self.omr_group_map else '.'))
                # total_mean = total_mean + painted_mean
                # pnum = pnum + 1
        # total_mean = total_mean / pnum
        # self.omr_recog_data['label'] = [1 if x > total_mean else 0
        #                                for x in self.omr_recog_data['bmean']]
        clu = KMeans(2)
        clu.fit(self.omr_recog_data['feature'])
        self.omr_recog_data['label'] = clu.predict(self.omr_recog_data['feature'])

    def get_recog_markcoord(self):
        # recog_data is error
        if len(self.omr_recog_data['label']) == 0:
            print('recog data not created yet!')
            return
        # num = 0
        xylist = []
        for coord, label in zip(self.omr_recog_data['coord'],
                                self.omr_recog_data['label']):
            if label == 1:
                xylist = xylist + [coord]
        return xylist

    # deprecated
    def get_result_dataframe(self):
        if len(self.omr_recog_data['label']) == 0:
            print('create dataframe fail:no recog data!')
            return pd.DataFrame(self.omr_recog_data)
        f = Tools.find_file(self.image_filename)
        rdf = pd.DataFrame({'card': [f] * len(self.omr_recog_data['label']),
                            'coord': self.omr_recog_data['coord'],
                            'label': self.omr_recog_data['label'],
                            'feat': self.omr_recog_data['feature'],
                            'group': self.omr_recog_data['group'],
                            'code': self.omr_recog_data['code'],
                            'mode': self.omr_recog_data['mode'],
                            })
        # 'label': self.omr_recog_data['label'],
        # 'bmean': self.omr_recog_data['bmean'],
        # set label2 sign to 1 for painted (1 at max mean value)
        # rdf['label3'] = rdf['bmean'].apply(lambda x:1 if x > self.omr_threshold else 0)
        # rdf['label3'] = rdf['label'] + rdf['label2']
        # rdf['label3'] = rdf['label3'].apply(lambda x:0 if x == 2 else x)
        if rdf.sort_values('feat', ascending=False).head(1)['label'].values[0] == 0:
            rdf['label'] = rdf['label'].apply(lambda x: 1 - x)
        # rdf.code = [rdf.code[i] if rdf.label[i] == 1 else '.' for i in range(len(rdf.code))]
        if rdf[rdf.group > 0].label.sum() == rdf[rdf.group > 0].count()[0]:
            rdf.label = 0
        rdf.loc[rdf.label == 0, 'code'] = '.'
        if not self.debug:
            r = rdf[rdf.group > 0][['card', 'group', 'coord', 'code', 'mode']]
            # r['codelen'] = r.code.apply(lambda s: len(s.replace('.', '')))
            return r.sort_values('group')
        self.omr_result_dataframe = rdf
        return rdf

    # new result dataframe
    def get_result_dataframe2(self):
        f = Tools.find_file(self.image_filename)

        # recog_data is error, return len=-1, code='XXX'
        if len(self.omr_recog_data['label']) == 0:
            if self.display:
                print('create dataframe fail: recog data is not created!')
            if self.group_result:
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
            return self.omr_result_dataframe
        # recog_data is ok
        rdf = pd.DataFrame({'coord': self.omr_recog_data['coord'],
                            'label': self.omr_recog_data['label'],
                            'feat': self.omr_recog_data['feature'],
                            'group': self.omr_recog_data['group'],
                            'code': self.omr_recog_data['code'],
                            'mode': self.omr_recog_data['mode'],
                            })
        if rdf.sort_values('feat', ascending=False).head(1)['label'].values[0] == 0:
            rdf['label'] = rdf['label'].apply(lambda x: 1 - x)
        # rdf.code = [rdf.code[i] if rdf.label[i] == 1 else '.' for i in range(len(rdf.code))]
        # all block painted !
        if rdf[rdf.group > 0].label.sum() == rdf[rdf.group > 0].count()[0]:
            rdf.label = 0
        rdf.loc[rdf.label == 0, 'code'] = '.'

        # create result dataframe
        outdf = rdf[rdf.group>0].sort_values('group')[['group','code']].groupby('group').sum()
        rs_codelen = 0
        rs_code = []
        group_str = ''
        invalid = 1
        if len(outdf) > 0:
            out_se = outdf['code'].apply(lambda s: ''.join(sorted(list(s.replace('.','')))))
            group_list = sorted(self.omr_group.keys())
            # print(out_se)
            # print(group_list)
            for g in group_list:
                if g in out_se.index:  # outdf.index:
                    # ts = outdf.loc[g, 'code'].replace('.', '')
                    # ts = ''.join(sorted(list(ts)))
                    if self.omr_group[g][4] in 'SM':
                        rs_code.append(self.omr_map_dict[out_se[g]])
                    else:
                        rs_code.append(out_se[g])
                    rs_codelen = rs_codelen + 1
                    group_str = group_str + str(g) + '_'
                else:
                    rs_code.append('@')
                    group_str = group_str + str(g) + '*_'
                    invalid = 0
            rs_code = ''.join(rs_code)
        else:
            rs_code = 'XXX'
            rs_codelen = -1
            invalid = 0

        # group result to dataframe: fname, len, group_str, result
        if self.group_result:
            # rdf['gstr'] = rdf.group.apply(lambda s: str(s) + ';')
            # rs_gcode = rdf[rdf.label == 1]['gstr'].sum()
            # result_group_no = rdf[(rdf.label == 1) & (rdf.group > 0)].sort_values('group')['gstr'].sum()
            '''
            if type(result_group_no) == str:
                if len(result_group_no) > 0:
                    result_group_no = result_group_no[:-1]  # remove ';' at end
                    if result_group_no == self.group_str:
                        result_group_no = '=='
                    # translate multi painting to map char
                    if len(result_group_no) > len(self.group_str):
                        g_list = result_group_no.split(';')
                        result_dict = {g:'' for g in g_list}
                        for g, r in zip(g_list, rs_code):
                            result_dict[g] += r
                        for g in result_dict:
                            if len(result_dict[g])>1:
                                result_dict[g] = self.omr_map_dict[result_dict[g]]
                '''
            # result_valid = 1 if rs_codelen == self.omr_code_valid_number else 0
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
        if not self.debug:
            rdf['card'] = f
            #self.omr_result_dataframe = rdf
            self.omr_debug_dataframe = rdf
            #return rdf
        return self.omr_result_dataframe

    # --- show omrimage or plot result data ---
    def plot_rawimage(self):
        plt.figure(1)
        plt.title(self.image_filename)
        plt.imshow(self.image_2d_matrix)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap)

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap)

    def plot_mark_omrimage(self):
        if type(self.mark_omriamge) != np.ndarray:
            self.get_mark_omrimage()
        plt.figure(4)
        plt.title('recognized - omr - region ' + self.image_filename)
        plt.imshow(self.mark_omriamge)

    def plot_recog_omrimage(self):
        if type(self.recog_omriamge) != np.ndarray:
            self.get_recog_omrimage()
        plt.figure(5)
        plt.title('recognized - omr - region' + self.image_filename)
        plt.imshow(self.recog_omriamge)

    def plot_omrblock_mean(self):
        plt.figure(6)
        plt.title(self.image_filename)
        data = self.omr_recog_data['mean'].copy()
        data.sort()
        plt.plot([x for x in range(len(data))], data)


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
        # ts = path_file
        # ts.replace('/', '\\')
        # p1 = ts.find('\\')
        # if p1 > 0:
        #    ts = ts[::-1]
        #    p1 = ts.find('\\')
        #    ts = ts[0: p1]
        #    ts = ts[::-1]
        # return ts

    @staticmethod
    def find_path(path_file):
        ts = Tools.find_file(path_file)
        return path_file.replace(ts, '')
    # class Tools end


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
