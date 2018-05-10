# -*- utf8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mg
import numpy as np
import pandas as pd
import cv2
import time
import os
import glob
import copy
from collections import Counter
from abc import ABCMeta, abstractclassmethod

def read128(
        code_type='128c',
        file_list=(),
        box_top=0, box_left=0, box_bottom=0, box_right=0,
        ratio_row=None, ratio_col=None,
        display=False
        ):
    """    
    :param code_type: 128a, 128b, 128c
    :param file_list: image files with barcode
    :param box_top: box top row in image
    :param box_left:   box left column in image
    :param box_bottom: box bottom row of image 
    :param box_right:  box right column in image
    :param display: display messages in processing
    :return: result, object of BarcodeReader128
    """
    if type(file_list) == str:
        file_list = [file_list]
    elif isinstance(file_list, list):
        file_list = file_list
    else:
        print('form is not valid type')
        return
    st = time.time()
    br = BarcodeReader128()
    # br.code_type = code_type
    br.get_barcode_dataframe(
        code_type=code_type,
        file_list=file_list,
        ratio_row=ratio_row, ratio_col=ratio_col,
        box_top=box_top, box_left=box_left, box_bottom=box_bottom, box_right=box_right,
        display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time() - st, (time.time()-st) / len(file_list)))
    return br


class BarcodeReader(object):
    def __init__(self):
        self.image_filenames = []

        # bar position in image
        self.box_top = 0
        self.box_bottom = 10000
        self.box_left = 0
        self.box_right = 10000

        # iamge processing parameters
        self.image_scan_scope = 12
        self.image_scan_step = 2
        self.image_scan_line_num = 5
        self.image_threshold_low = 10
        self.image_threshold_high = 110
        self.image_threshold_step = 6
        self.image_use_ratio = False
        self.image_ratio_row = 1
        self.image_ratio_col = 1

        # image data in procedure
        self.image_raw = None
        self.image_cliped = None
        self.image_gradient = None
        self.image_blurred = None
        self.image_closed = None
        self.image_bar = None
        self.image_bar01 = None
        self.image_mid_row = 0

        # bar data in procedure
        self.bar_pwlist_dict = {}
        self.bar_pwglist_dict = {}
        self.bar_bscode_dict = {}
        self.bar_codecount_list = {}
        self.bar_collect_codecount_list = []
        self.bar_codelist_candidate_list = []
        self.bar_codelist_length = 0

        # result code and other data
        self.result_code = ''
        self.result_codelist = []
        self.result_codelist_validity = []
        self.result_code_possible = []
        self.result_code_valid = False
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = 0
        self.result_dataframe = None

        # tools
        self.checker = BarCheckerFactory.create('128')

    def set_image_files(self, file_list):
        self.image_filenames = file_list

    def set_image_box(self, box_top=None, box_bottom=None, box_left=None, box_right=None):
        self.box_top = box_top
        self.box_bottom = box_bottom
        self.box_left = box_left
        self.box_right = box_right

    def set_image_ratio(self, ratio_row=None, ratio_col=None):
        if ratio_row is not None:
            self.image_ratio_row = ratio_row
        if ratio_col is not None:
            self.image_ratio_col = ratio_col

    # get image_raw from filename
    def get_image_from_file(self, image_file, display=False):
        # read image data from image file
        if (type(image_file) != str) or (image_file == ''):
            if display:
                print('no image file given!')
            return False
        else:
            if not os.path.isfile(image_file):
                print('not found file: %s' % image_file)
                return False
        # read image,  from self.image_filenames
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_raw = image
        return True

    # get image_bar from self.image_raw
    def get1_image_bar(self, image_data=None, display=False):
        # check self.image_raw
        if image_data is None:
            if display:
                print('image_data is None!')
            return False

        self.image_cliped = image_data[self.box_top: self.box_bottom+1,
                                       self.box_left: self.box_right+1]

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        gradx = cv2.Sobel(self.image_cliped, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grady = cv2.Sobel(self.image_cliped, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradx, grady)
        self.image_gradient = cv2.convertScaleAbs(gradient)

        self.image_blurred = cv2.blur(gradient, (9, 9))

        (_, thresh) = cv2.threshold(self.image_blurred, 225, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        self.image_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.image_closed = cv2.erode(self.image_closed, None, iterations=3)
        self.image_closed = cv2.dilate(self.image_closed, None, iterations=5)
        # plt.imshow(self.closed)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        # get 8UC1 format image
        img = cv2.normalize(self.image_closed,
                            None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8UC1)
        (_, contours, __) = cv2.findContours(
                                img.copy(),
                                mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            # compute the rotated bounding box of the largest contour
            # get bar image from box area
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv2.boxPoints(rect))
            # print(box)
            left, top, right, bottom = \
                min(box[:, 0]) - 15 if min(box[:, 0]) > 15 else min(box[:, 0]), \
                min(box[:, 1]) - 15 if min(box[:, 1]) > 15 else min(box[:, 1]), \
                max(box[:, 0]) + 15, \
                max(box[:, 1]) + 15
            self.image_bar = self.image_cliped[top:bottom, left:right]
            self.result_barimage_found = True
        else:
            self.image_bar = self.image_cliped.copy()
            self.result_barimage_found = False

        # image_bar is empty instead of using image_cliped
        # if (self.image_bar.shape[0] == 0) | (self.image_bar.shape[1] == 0):
            # print('no bar image found!')
        #    self.image_bar = self.image_cliped.copy()
        #    self.result_barimage_found = False

        # get mid row loc
        cl = (255-self.image_bar).sum(axis=1)
        cl_mean = cl.mean()
        cl_peak = np.where(cl > cl_mean*1.62)[0]
        if len(cl_peak) > 0:
            self.image_mid_row = int((cl_peak[0] + cl_peak[-1]) / 2)
        else:
            self.image_mid_row = int(self.image_bar.shape[0] / 2)
        return True

    def show_raw_iamge(self):
        plt.figure('raw image')
        plt.imshow(self.image_raw)

    def show_bar_iamge(self):
        plt.figure('gray bar image')
        plt.imshow(self.image_bar)

    def show_bar_iamge01(self):
        plt.figure('binary bar image')
        plt.imshow(self.image_bar01)

    def show_bar_wslist_list(self):
        for k in self.bar_pwglist_dict:
            print('{0}:{1}'.format(k, self.bar_pwglist_dict[k]))

    def show_bar_bscode_list(self):
        for k in self.bar_bscode_dict:
            print('{0}:{1}'.format(k, self.bar_bscode_dict[k]))


class BarcodeReader128(BarcodeReader):

    def __init__(self):
        super(BarcodeReader128, self).__init__()

        b128a = BarcodeTable128('128a')
        self.table_128a, self.table_128ased = b128a.code_table, b128a.code_table_sed
        b128b = BarcodeTable128('128b')
        self.table_128b, self.table_128bsed = b128b.code_table, b128b.code_table_sed
        b128c = BarcodeTable128('128c')
        self.table_128c, self.table_128csed = b128c.code_table, b128c.code_table_sed

        self.code_type = '128c'
        self.code_form = []

    def get_barcode_dataframe(
            self,
            code_type=None,
            file_list=None,
            box_top=None, box_left=None, box_bottom=None, box_right=None,
            ratio_row=None, ratio_col=None,
            display=False
            ):

        # set iamge files to read
        if type(file_list) == list:
            self.image_filenames = file_list
        elif type(file_list) == str:
            self.image_filenames = [file_list]
        else:
            print('invalid files {}'.format(file_list))
        # set code_type
        if isinstance(code_type, str):
            self.code_type = code_type

        # read barcode to dataframe from imagefiles(file_list)
        for i, _file in enumerate(file_list):
            self.get_barcode_from_image_file(
                image_file=_file,
                box_top=box_top, box_bottom=box_bottom, box_left=box_left, box_right=box_right,
                ratio_row=ratio_row, ratio_col=ratio_col,
                display=display
                )
            if i > 0:
                self.result_dataframe = \
                    self.result_dataframe.append(
                        pd.DataFrame({
                            'file': [BarcodeUtil.find_file_from_pathfile(_file)],
                            'code': [self.result_code],
                            'codelist': [self.result_codelist],
                            'validity': [self.result_codelist_validity],
                            'valid': [self.result_code_valid],
                            'steps': [self.result_detect_steps],
                            'found': [self.result_barimage_found],
                            'fill': [self.result_fill_loss]
                            }, index=[i]))
            else:
                self.result_dataframe = \
                    pd.DataFrame({
                        'file': [BarcodeUtil.find_file_from_pathfile(_file)],
                        'code': [self.result_code],
                        'codelist': [self.result_codelist],
                        'validity': [self.result_codelist_validity],
                        'valid': [self.result_code_valid],
                        'steps': [self.result_detect_steps],
                        'found': [self.result_barimage_found],
                        'fill': [self.result_fill_loss]
                        }, index=[i])
            print(i,
                  BarcodeUtil.find_file_from_pathfile(_file),
                  self.result_code,
                  self.result_codelist,
                  self.result_code_valid
                  )

    def get_barcode_from_image_file(
            self,
            image_file=None,
            box_top=None, box_bottom=None, box_left=None, box_right=None,
            ratio_row=None, ratio_col=None,
            image_threshold_low=None,
            image_threshold_high=None,
            image_threshold_step=None,
            image_scan_scope=None,
            image_scan_step=None,
            image_scan_line_num=None,
            display=False
            ):
        # initiate result data
        self.result_code_valid = False
        self.result_code = ''
        self.result_codelist = []
        self.result_code_possible = []
        self.result_codelist_validity = []
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = False

        # read image to self.image_raw from image_file
        if not self.get_image_from_file(image_file=image_file, display=display):
            if display:
                print('no image file {}'.format(image_file))
            return

        self.get_barcode_from_image_data(
            image_data=self.image_raw,
            box_top=box_top, box_bottom=box_bottom, box_left=box_left, box_right=box_right,
            ratio_row=ratio_row, ratio_col=ratio_col,
            image_threshold_low=image_threshold_low,
            image_threshold_high=image_threshold_high,
            image_threshold_step=image_threshold_step,
            image_scan_scope=image_scan_scope,
            image_scan_step=image_scan_step,
            image_scan_line_num=image_scan_line_num,
            display=display
            )
        return
        # end get_bar_from_file

    def get_barcode_from_image_data(
            self,
            code_type=None,
            image_data=None,
            box_top=None, box_left=None, box_bottom=None, box_right=None,
            ratio_row=None, ratio_col=None,
            image_threshold_low=None, image_threshold_high=None, image_threshold_step=None,
            image_scan_scope=None, image_scan_step=None, image_scan_line_num=None,
            display=False
            ):
        # check input para
        if type(code_type) == str:
            if code_type not in ['128a', '128b', '128c']:
                print('invalid code type={}'.format(code_type))
                return

        # set bar area
        if box_top is not None:
            self.box_top = box_top
        if box_left is not None:
            self.box_left = box_left
        if box_bottom is not None:
            self.box_bottom = box_bottom
        if box_right is not None:
            self.box_right = box_right

        # set other para
        if type(image_threshold_low) == int:
            self.image_threshold_low = image_threshold_low
        if type(image_threshold_high) == int:
            self.image_threshold_high = image_threshold_high
        if type(image_threshold_step) == int:
            self.image_threshold_step = image_threshold_step
        if type(image_scan_scope) == int:
            self.image_scan_scope = image_scan_scope
        if type(image_scan_step) == int:
            self.image_scan_step = image_scan_step
        if type(image_scan_line_num) == int:
            self.image_scan_line_num = image_scan_line_num

        # initiate proc and result var
        self.bar_pwlist_dict = {}
        self.bar_pwglist_dict = {}
        self.bar_bscode_dict = {}
        self.bar_codecount_list = {}
        self.bar_collect_codecount_list = []
        self.bar_codelist_length = 0
        self.bar_codelist_candidate_list = []
        self.result_codelist = []
        self.result_code = ''
        self.result_code_valid = False
        self.result_code_possible = []
        self.result_codelist_validity = []
        self.result_detect_steps = 0
        self.result_barimage_found = False
        self.result_fill_loss = False
        get_result = False

        # get bar image
        if not self.get1_image_bar(image_data=image_data, display=display):
            if display:
                print('fail to extract bar from raw image!')
            return False

        # first check barcode
        self.result_detect_steps += 1
        if display:
            print('---the first check with raw bar image---')
        self.get2_codelist_from_image(code_type=code_type, display=display)
        if self.get3_result_code(display=display):
            get_result = True

        # second check barcode by amplify image
        if (not get_result) | ('*' in ''.join(self.result_codelist)):
            self.result_detect_steps += 1
            if display:
                print('---the second check with amplified bar image({0}, {1})---'.format(1.15, 1.25))
            self.image_bar = BarcodeUtil.image_amplify(self.image_bar, ratio_row=1.15, ratio_col=1.25)
            self.get2_codelist_from_image(code_type=code_type, display=display)
            if self.get3_result_code(display=display):
                get_result = True

        # the third check by amplify image
        if (not get_result) | ('*' in ''.join(self.result_codelist)):
            self.result_detect_steps += 1
            if display:
                print('---the third check with amplified bar image({0}, {1})---'.format(1.2, 1.5))
            self.image_bar = BarcodeUtil.image_amplify(self.image_bar, ratio_row=1.2, ratio_col=1.5)
            self.get2_codelist_from_image(code_type=code_type, display=display)
            if self.get3_result_code(display=display):
                get_result = True

        # the 3+1 check by amplify image
        if (not get_result) & ((ratio_row is not None) | (ratio_col is not None)):
            ratio_row = 1 if ratio_row is None else ratio_row
            ratio_col = 1 if ratio_col is None else ratio_col
            self.result_detect_steps += 1
            if display:
                print('---the third+ check with amplified bar image({0}, {1})---'.format(ratio_row, ratio_col))
            self.image_bar = BarcodeUtil.image_amplify(self.image_bar, ratio_row=ratio_row, ratio_col=ratio_col)
            self.get2_codelist_from_image(code_type=code_type, display=display)
            if self.get3_result_code(display=display):
                get_result = True

        # the fourth check by filling
        if len(self.result_codelist) > 3:
            if (not get_result) & self.result_codelist[-2].isdigit():
                self.result_detect_steps += 1
                if display:
                    print('---fourth check with filling---')
                self.get4_result_by_fill_lowvalidity(display=display)
                # get result code by filling star else result0
                # self.get_result_code_from_candidate_by_filling(display)

        if display:
            print('--' * 60,
                  '\n result_code = {0} '
                  '\npossiblecode = {1} '
                  '\n check_times = {2}'
                  '\n    validity = {3}'.
                  format(self.result_code,
                         self.result_code_possible,
                         self.result_detect_steps,
                         self.result_codelist_validity))

        return True
        # end get_barcode

    # get result code from self.bar_codelist_candidate_list
    def get3_result_code(self, display=False):
        self.result_code_valid = False
        # select code if no '*' in code, checkcode
        for code_list in self.bar_codelist_candidate_list:
            code_len = len(code_list)
            # invalid codelist length
            if code_len <= 3:
                continue
            # verify checkcode and return result
            check_code = code_list[code_len - 2]
            if (check_code.isdigit()) & ('*' not in ''.join(code_list[1:code_len-1])):
                if display:
                    print('no loss candidate:', code_list)
                # if self.get_codelist_check(code_list[1:code_len - 2], check_code, display):
                if self.checker.check_codelist(code_list, self.code_type)[0]:
                    self.cmp_code_from_codelist(codelist=code_list)
                    return True
        # computing max score codelist from candidate if no valid codelist
        result_codelist0 = self.get_maxscore_codelist_from_candidate(self.bar_codelist_candidate_list)
        if len(result_codelist0) > 3:
            self.cmp_code_from_codelist(codelist=result_codelist0)
        return self.result_code_valid

    def cmp_code_from_codelist(self, codelist):
        # self.result_code_valid = self.get_codelist_check(codelist[1:-2], codelist[-2])
        self.result_code_valid = \
            self.checker.check_codelist(codelist=codelist, code_type=self.code_type)[0]
        self.result_code = \
            ''.join([s for s in codelist[1:-2]
                     if s not in ['CodeA', 'CodeB', 'CodeC', 'FNC1', 'FNC4', 'SHIFT']])
        self.result_codelist = codelist
        self.result_codelist_validity = self.cmp_codelist_validity(codelist)
        return

    def cmp_codelist_validity(self, codelist):
        # print(self.bar_collect_codecount_list, codelist)
        if len(self.bar_collect_codecount_list) != len(codelist):
            return []
        cdvalidity = []
        for ci, cd in enumerate(codelist):
            if cd in self.bar_collect_codecount_list[ci]:
                cdvalidity.append(self.bar_collect_codecount_list[ci][cd])
            else:
                cdvalidity.append(0)
        # sm = sum(cdvalidity)
        sm = np.mean(cdvalidity)
        if sm > 100:
            # sm = sum(cdvalidity)/len(cdvalidity)
            return [eval('{:.4}'.format(v / sm)) for v in cdvalidity]
        else:
            return [eval('{:.4}'.format(v/100)) for v in  cdvalidity]

    def get4_result_by_fill_lowvalidity(self, display=False):
        if display:
            print('check validity:{}'.format(self.result_codelist_validity))
        if len(self.result_codelist_validity) == 0:
            return False
        lowvalidity = min(self.result_codelist_validity)
        if lowvalidity > 0.3:
            if display:
                print('min validity({})>0.3, abandon to fill'.format(lowvalidity))
            return False
        loc = self.result_codelist_validity.index(lowvalidity)
        if 0 < loc < len(self.result_codelist) - 2:  # think, not include checkcode
            newcodelist = self.result_codelist.copy()
            newcodelist[loc] = '**'
            filled_codelists = self.bar_128_fill_loss(newcodelist[1:-2], self.result_codelist[-2])
            if len(filled_codelists) > 0:
                if len(filled_codelists) > 1:
                    self.result_code_possible = filled_codelists[1:]
                codelist = [self.result_codelist[0]] + filled_codelists[0] + self.result_codelist[-2:]
                self.cmp_code_from_codelist(codelist=codelist)
                self.result_fill_loss += 1
                if display:
                    print('finish filling:{0} '
                          '\n          from:{1}'
                          '\n   result code:{2}'.
                          format(self.result_codelist, newcodelist, self.result_code))
                return True
        return False

    # select codelist with big score(occurrence times)
    def get_maxscore_codelist_from_candidate(self, codelist_list):
        if len(codelist_list) == 0:
            return []
        result_codelist_score = [0 for _ in codelist_list]
        # maxscore = 0
        for ci, cl in enumerate(codelist_list):
            if len(cl) != len(self.bar_collect_codecount_list):
                continue
            score = 0
            for di, dk in enumerate(cl):
                if dk in self.bar_collect_codecount_list[di]:
                    score += self.bar_collect_codecount_list[di][dk]
            result_codelist_score[ci] = score
            # maxscore = max(maxscore, score)
        maxscore = max(result_codelist_score)
        return codelist_list[result_codelist_score.index(maxscore)]

    # get cnadidate_codelist_list
    # by get_pwlist, get_codecount, get_collect_codecount
    def get2_codelist_from_image(self, code_type, display=False):

        # scan for gray_threshold
        # barimage-->pwlist,wslist-->codecount_list-->collect_codecount_list
        for th_gray in range(self.image_threshold_low,
                             self.image_threshold_high,
                             self.image_threshold_step):
            # get pwlist
            self._get2x1_pwlist_from_barimage(gray_shift=th_gray)
            # get codecount
            if self._get2x2_codecount_from_pwlist(code_type=code_type,
                                                  th_gray=th_gray,
                                                  display=display):
                # collect codecount
                self._get2x3_collect_codecount_list(codecount_list=self.bar_codecount_list)
        if display:
            print('collect codecount:{}'.format(self.bar_collect_codecount_list))

        # remove empty element in tail
        if len(self.bar_collect_codecount_list) < 3:
            return
        stop_loc = len(self.bar_collect_codecount_list) - 1
        for i, c in enumerate(self.bar_collect_codecount_list[2:], start=2):
            # set i as new stop loc
            if ('Stop' in c) & ('Stop' in self.bar_collect_codecount_list[stop_loc]):
                if (c['Stop'] > 50) & (self.bar_collect_codecount_list[stop_loc]['Stop'] < 10):
                    stop_loc = i
                    break
            if 'Stop' in self.bar_collect_codecount_list[stop_loc]:
                if self.bar_collect_codecount_list[stop_loc]['Stop'] > 30:
                    continue
            # go back 1 when current count is 0
            if (len(c) == 0) & ('Stop' in self.bar_collect_codecount_list[i - 1]):
                if self.bar_collect_codecount_list[i-1]['Stop'] > 50:   # think 50
                    stop_loc = i-1
                    break
            # go back 1 when current count less than 10
            if ('Stop' in c) & ('Stop' in self.bar_collect_codecount_list[i - 1]):
                if (c['Stop'] < 10) & (self.bar_collect_codecount_list[i - 1]['Stop'] > 100):
                    stop_loc = i - 1
                    break
            # go back 1 when current less than last(i-1)
            if ('Stop' in c) & ('Stop' in self.bar_collect_codecount_list[i - 1]):
                if c['Stop'] < self.bar_collect_codecount_list[i - 1]['Stop']:
                    stop_loc = i - 1
                    break
        self.bar_collect_codecount_list = self.bar_collect_codecount_list[0:stop_loc+1]

        # remove no 'Start', 'Stop' element in head/tail code_dict
        meancount = sum([max(list(d.values()))
                        if len(d) > 0 else 0 for d in self.bar_collect_codecount_list]) / \
            len(self.bar_collect_codecount_list)
        if 'Start' in ''.join(list(self.bar_collect_codecount_list[0].keys())):
            self.bar_collect_codecount_list[0] = {k: int(meancount) for k in self.bar_collect_codecount_list[0]
                                                  if 'Start' in k}    # think of 'StartA B C'
        if 'Stop' in ''.join(list(self.bar_collect_codecount_list[-1].keys())):
            self.bar_collect_codecount_list[-1] = {'Stop': int(meancount)}    # think 'Stop'

        # prunning redundant node: 'Stop', 'CodeA...C'
        collect2 = []
        for ci, cd in enumerate(self.bar_collect_codecount_list):
            if 0 < ci < len(self.bar_collect_codecount_list)-1:  # not head, tail, more than 2 elenments
                if 'Stop' in cd:
                    del cd['Stop']
                for c0 in ['StartA', 'StartB', 'StartC']:
                    if c0 in cd:
                        del cd[c0]
                for c0 in ['CodeA', 'CodeB', 'CodeC']:
                    if c0 in cd:
                        if (cd[c0] < 10) & (max([cc[c0] for cc in self.bar_collect_codecount_list if c0 in cc]) > 50):
                            del cd[c0]
            collect2.append(cd)
        self.bar_collect_codecount_list = collect2

        if display:
            print('reuslt collection:{}'.format(self.bar_collect_codecount_list))

        # get candidate_codelist from collect_codecount_list
        self.bar_codelist_candidate_list = self._get2x4x2_candidate_codelist()
        return

    def _get2x1_pwlist_from_barimage(self, gray_shift=60):
        # get pwlist from image_bar
        #   with para: self.image_detect_win_high, scan_scope, gray_shift

        # get binary image
        img = 255 - self.image_bar.copy()
        th = img.mean() + gray_shift
        img[img < th] = 0
        img[img > 0] = 1
        self.image_bar01 = img

        # get bar bar&space width list
        mid_loc = self.image_mid_row
        for _line in range(-self.image_scan_scope,
                           self.image_scan_scope,
                           self.image_scan_step):
            row = mid_loc + _line
            img_line = np.around(self.image_bar01[row: row + self.image_scan_line_num, :].sum(axis=0) /
                                 self.image_scan_line_num, decimals=0)
            # trip head & tail 0
            for j in range(len(img_line)):
                if img_line[j] == 1:
                    img_line = img_line[j:]
                    break
            for j in range(len(img_line) - 1, 0, -1):
                if img_line[j] == 1:
                    img_line = img_line[:j + 1]
                    break
            # get pwlist
            bs_wid_list, lastc, curwid = [], img_line[0], 1
            for cc in img_line[1:]:
                if cc != lastc:
                    bs_wid_list.append(curwid)
                curwid = curwid + 1 if cc == lastc else 1
                lastc = cc
            bs_wid_list.append(curwid)
            self.bar_pwlist_dict[(gray_shift, _line)] = bs_wid_list
        return

    # get codeCountDict from pwlist for a fixed th_gray and all scanning line
    # from bar_bspixel_list_dict[th_gray, -scope:scope]
    def _get2x2_codecount_from_pwlist(self, code_type=None, th_gray=20, display=False):
        mlines_code_dict = dict()
        for line_no in range(-self.image_scan_scope,
                             self.image_scan_scope,
                             self.image_scan_step):
            result = self._get2x2x1_decode_from_pwlist(self.bar_pwlist_dict[(th_gray, line_no)],
                                                       th_gray=th_gray,
                                                       line_no=line_no,
                                                       code_type=code_type,
                                                       display=display)
            if len(result) > 0:
                mlines_code_dict[line_no] = result

        # get codecount from result_dict
        if len(mlines_code_dict) > 0:
            max_len = max([len(mlines_code_dict[x]) for x in mlines_code_dict])
        else:
            return False

        self.bar_codecount_list = [{} for _ in range(max_len)]
        for line_no in range(-self.image_scan_scope,
                             self.image_scan_scope,
                             self.image_scan_step):
            # not valid line or invalid bs list(no data items)
            if (line_no not in mlines_code_dict) or (len(mlines_code_dict[line_no]) < 4):
                continue
            for di, dc in enumerate(mlines_code_dict[line_no]):
                if dc.isdigit() or (dc in ['StartC', 'Stop', 'CodeB']):
                    if dc in self.bar_codecount_list[di]:
                        self.bar_codecount_list[di][dc] += 1
                    else:
                        self.bar_codecount_list[di][dc] = 1

        if display:
            print('scan th_gray={}:'.format(th_gray), self.bar_codecount_list)

        # abandon list with 4 or more empty code items
        if sum([0 if len(d) > 0 else 1 for d in self.bar_codecount_list]) > 3:
            return False
        return True

    # decode from pwlist by bscode:code
    # use formula: max(barnum)==4 in 128, think
    def _get2x2x1_decode_from_pwlist(self, pwlist=(),
                                     th_gray=0,
                                     line_no=0,
                                     code_type=None,
                                     display=False):
        if (len(pwlist) - 1) % 6 > 0:
            # if display:
                # print('pwlist length error: gray_shift={}, scan_line={}, length={}'.
                #      format(th_gray, line_no, len(pwlist)))
            return []

        # seek code
        wid_list = []
        for i in range(0, len(pwlist), 6):
            if (i + 8) < len(pwlist):
                wid_list.append(pwlist[i:i + 6])
            else:
                wid_list.append(pwlist[i:])
                break
        self.bar_pwglist_dict[(self.result_detect_steps, th_gray, line_no)] = wid_list

        bscode_list = []
        for s in wid_list:
            sw = sum(s)
            si = ''
            bar_unit_len = 11 if len(s) == 6 else 13
            for r in s:
                bv = round(bar_unit_len * r / sw)
                bv = 1 if bv == 0 else 4 if bv > 4 else bv
                si = si + str(bv)
            bscode_list.append(si)
        self.bar_bscode_dict[(th_gray, line_no)] = bscode_list

        # select codeset
        if code_type is not None:
            codetype1 = {'128a': 'A', '128b': 'B', '128c': 'C'}.get(code_type.lower(), '*')
        else:
            codetype1 = {'211412': 'A', '211214': 'B', '211232': 'C'}.get(bscode_list[0], '*')
        if False:  # codetype1 == '*':
            if display:
                print('pwlist startcode error: gray_shift={0}, scan_line={1}, startbs={2}'.format
                      (th_gray, line_no, bscode_list))
            # default 128c?
            codetype1 = 'C'

        # deal with mixing encoding among 128a, 128b, 128c
        codeset = 0  # o:no, 1:128a, 2:128b, 3:128c
        code_dict = self.table_128a if codetype1 == 'A' else \
            self.table_128b if codetype1 == 'B' else self.table_128c
        result = []
        for bs in bscode_list:
            dc = code_dict[bs] if bs in code_dict else \
                 '*' if (code_type == '128c') & (codeset == 2) else '**'
            result.append(dc)
            # use CodeB only 1 time in 128c
            if (codetype1 == 'C') & (codeset != 3):
                dc = 'CodeC'
            if dc in ['StartA', 'CodeA']:
                code_dict = self.table_128a
                codeset = 1
            elif dc in ['StartB', 'CodeB']:
                code_dict = self.table_128b
                codeset = 2
            elif dc in ['StartC', 'CodeC']:
                code_dict = self.table_128c
                codeset = 3
        # decode check code in 128C
        if len(result) > 3:
            if (codetype1 == 'C') & (result[-2] in ['CodeA', 'CodeB', 'FNC1']):
                result[-2] = {'CodeB': '100', 'CodeA': '101', 'FNC1': '102'}[result[-2]]
        else:
            return []

        return result

    # get code from pwlist by bscode_sed-->code
    def _get2x2x2_decode_from_pwlist(self, pwlist=(), th_gray=0, line_no=0,
                                     code_type=None, display=False):
        if (len(pwlist) - 1) % 6 > 0:
            # if display:
                # print('pwlist length error: gray_shift={}, scan_line={}, length={}'.
                #      format(th_gray, line_no, len(pwlist)))
            return []

        # seek code
        wid_list = []
        for i in range(0, len(pwlist), 6):
            if (i + 8) < len(pwlist):
                wid_list.append(pwlist[i:i + 6])
            else:
                wid_list.append(pwlist[i:])
                break
        self.bar_pwglist_dict[(self.result_detect_steps, th_gray, line_no)] = wid_list

        # calculate similar edge distance
        bs_str = []
        for w in wid_list:
            sw = sum(w)
            si = ''
            bar_unit_len = 11 if len(w) == 6 else 13
            for ri in range(len(w)-1):
                si = si + str(round(bar_unit_len * (w[ri]+w[ri+1]) / sw))
            bs_str.append(si)

        # select codeset, default is 128c
        if code_type is not None:
            codetype1 = {'128a': 'A', '128b': 'B', '128c': 'C'}.get(code_type.lower(), 'C')
        else:
            codetype1 = {'32553': 'A', '32335': 'B', '32355': 'C'}.get(bs_str[0], 'C')

        # deal with mixing encoding among 128a, 128b, 128c
        codeset = 0  # o:no, 1:128a, 2:128b, 3:128c
        if codetype1 == 'A':
            code_dict = self.table_128ased
        elif codetype1 == 'B':
            code_dict = self.table_128bsed
        else:
            code_dict = self.table_128csed
        result = []
        for bs in bs_str:
            if bs in code_dict:
                dc = code_dict[bs]
            elif (codetype1 == 'C') & (codeset != 3):   # find CodeA/CodeB in CodeC
                dc = '*'
            else:
                dc = '**'
            result.append(dc)
            # use CodeB only 1 time in 128c
            if (codetype1 == 'C') & (codeset != 3):
                dc = 'CodeC'
            if dc in ['StartA', 'CodeA']:
                code_dict = self.table_128ased
                codeset = 1
            elif dc in ['StartB', 'CodeB']:
                code_dict = self.table_128bsed
                codeset = 2
            elif dc in ['StartC', 'CodeC']:
                code_dict = self.table_128csed
                codeset = 3
        if len(result) > 3:
            # set check code to serial no
            if (codetype1 == 'C') & (result[-2] in ['CodeA', 'CodeB', 'FNC1']):
                result[-2] = {'CodeB': '100', 'CodeA': '101', 'FNC1': '102'}[result[-2]]
            if (codetype1 == 'B') & (result[-2] in ['CodeA', 'FAN4', 'FNC1']):
                result[-2] = {'FAN4': '100', 'CodeA': '101', 'FNC1': '102'}[result[-2]]
            if (codetype1 == 'A') & (result[-2] in ['FAN4', 'CodeB', 'FNC1']):
                result[-2] = {'CodeB': '100', 'FAN4': '101', 'FNC1': '102'}[result[-2]]
        else:
            return []

        return result

    # get self.bar_collect_codecount_list from scanline codecount_list(self.bar_codecount_list)
    def _get2x3_collect_codecount_list(self, codecount_list):
        if len(self.bar_collect_codecount_list) == 0:
            # first time to initiate
            self.bar_collect_codecount_list = codecount_list
            self.bar_codelist_length = len(codecount_list)
        else:
            # others add
            self.bar_codelist_length = (self.bar_codelist_length + len(codecount_list))/2
            for i, dc in enumerate(codecount_list):
                if i < len(self.bar_collect_codecount_list):
                    for kc in dc:
                        if kc in self.bar_collect_codecount_list[i]:
                            self.bar_collect_codecount_list[i][kc] += dc[kc]
                        else:
                            self.bar_collect_codecount_list[i][kc] = dc[kc]
                else:
                    self.bar_collect_codecount_list.append(dc)
        return

    # get candidate_codelist from collent_codecount_list
    def _get2x4x1_candidate_codelist(self):
        codecount_list = self.bar_collect_codecount_list

        # return [] if empty codedict number > 3
        if sum([1 for d in codecount_list if len(d) == 0]) > 3:
            self.result_code = ''
            return []

        count_order_list = [sorted([(d[k], k) for k in d if len(k) > 0]
                                   if len(d) > 0 else [(1, '**')], reverse=True)
                            for d in codecount_list]
        count_list_len = [len(d)-1 for d in count_order_list]
        count_list_var = [0 if len(d) > 1 else -1 for d in codecount_list]
        code_len = len(codecount_list)
        result_list = []

        # single code list only
        if sum(count_list_var) == -code_len:
            # remove code after 'Stop'
            code_list = []
            for di, d in enumerate(codecount_list):
                if len(d) == 0:
                    code_list.append('**')
                else:
                    dc = list(d.keys())[0]
                    if (dc == 'Stop') & (di < len(codecount_list)-1):
                            code_list.append('**')
                    else:
                        code_list.append(dc)
            result_list = [(1, code_list)]
            loop = False
        else:
            loop = True

        # select candidate codelist by travers collect codecount
        while loop:
            # print(count_list_var)
            for j in range(code_len):
                if count_list_var[j] < 0:
                    continue
                # create new list
                code_list_score = sum([count_order_list[vi][v][0] for vi, v in enumerate(count_list_var)])
                code_list_0 = [count_order_list[vi][v][1] for vi, v in enumerate(count_list_var)]
                # remove code after 'Stop'
                code_list = []
                for di, d in enumerate(code_list_0):
                    if len(d) == 0:
                        code_list.append('**')
                    else:
                        if (d == 'Stop') & (di < len(code_list_0)-1):
                            code_list.append('**')
                        else:
                            code_list.append(d)
                result_list.append((code_list_score, code_list))
                # record times selected
                if count_list_var[j] < count_list_len[j]:
                    count_list_var[j] += 1
                    break
                else:
                    loop = False
                    for n in range(j+1, code_len-1):
                        if count_list_var[n] < 0:
                            continue
                        if count_list_var[n] < count_list_len[n]:
                            count_list_var[j] = 0
                            count_list_var[n] += 1
                            # low digit to 0
                            for k in range(n):
                                if count_list_var[k] >= 0:
                                    count_list_var[k] = 0
                            loop = True
                            break
                    break
        # set order according to score in result_lists
        if len(result_list) > 1:
            result_list_sort = sorted(result_list, reverse=True)
            result_lists = [r[1] for r in result_list_sort]
        else:
            result_lists = [result_list[0][1]]

        # set code='*' after CodeB if not digits, code=0-9 if int-16 in [0,9]
        result_list_codeb = []
        for cl in result_lists:
            cl1 = cl.copy()
            codeb = 0
            for j in range(1, len(cl)-2):
                if codeb == 1:
                    codeb = 0
                    if not cl[j].isdigit():
                        cl1[j] = '*'
                    elif (len(cl[j]) == 2) & (0 <= int(cl[j])-16 <= 9):
                        cl1[j] = str(int(cl[j])-16)
                    else:
                        cl1[j] = '*'
                elif cl[j] == 'CodeB':
                    codeb = 1
            result_list_codeb.append(cl1)

        return result_list_codeb

    # get candidate_codelist from collent_codecount_list
    def _get2x4x2_candidate_codelist(self):
        # get candidate_list from collect_list
        result_lists = self._get2x4x2a_best_codelist_from_collect()
        # just for 128c
        # set code='*' after CodeB if not digits, code=0-9 if int-16 in [0,9]
        if self.code_type.lower() == '128c':
            result_lists2 = []
            for cl in result_lists:
                if cl[-3] in ['CodeA', 'CodeB', 'CodeC']:
                    cl.append('Stop')
                for j in range(2, len(cl)-1):
                    if cl[j-1] == 'CodeB':
                        if not cl[j].isdigit():
                            cl[j] = '*'
                        elif (len(cl[j]) == 2) & (0 <= int(cl[j])-16 <= 9):
                            cl[j] = str(int(cl[j])-16)
                    if j == len(cl)-2:
                        if not cl[j].isdigit():
                            cl[j] = '**'
                result_lists2.append(cl)
            return result_lists2
        return result_lists

    def _get2x4x2a_best_codelist_from_collect(self):
        codecount_list = copy.deepcopy(self.bar_collect_codecount_list)
        result_lists = []
        # return [] if empty codecount number > 3
        if sum([1 for d in codecount_list if len(d) == 0]) > 3:
            return []

        # print('init cd:',codecount_list)
        # select maxcount codelist
        check_valid = False
        while True:
            # print('select--', codecount_list)
            loop_break = True
            select_codelist = []
            for ci, cd in enumerate(codecount_list):
                if len(cd) == 0:
                    select_codelist.append('**')
                    continue
                mv = max(list(cd.values()))
                md = [d for d in cd if cd[d] == mv]
                select_codelist.append(md[0])
                # print('to pop--', md, md[0], len(md))
                if len(md) > 1:
                    codecount_list[ci].pop(md[0])
                    loop_break = False
            result_lists.append(select_codelist)
            # if self.get_codelist_check(select_codelist[1:-2], select_codelist[-2]):
            if self.checker.check_codelist(codelist=select_codelist, code_type=self.code_type)[0]:
                check_valid = True
                break
            if loop_break:
                break

        # select other checkcode if check does not pass
        result_lists2 = result_lists.copy()     # avoid endless loop
        ckdict = codecount_list[-2]
        if (not check_valid) & (len(ckdict) > 0):
            for code_list in result_lists:
                # print('selected codelist: ', code_list)
                if '*' not in ''.join(code_list):
                    for d in ckdict:
                        # if self.get_codelist_check(code_list[1:-2], d):
                        newlist = code_list[0:-2]+[d]+[code_list[-1]]
                        if self.checker.check_codelist(codelist=newlist, code_type=self.code_type)[0]:
                            result_lists2.append(newlist)
            return result_lists2
        return result_lists

    # deprecated now
    # select best code from self.bar_candidate_codelist_list
    def get_result_code_from_candidate_by_filling(self, display):
        for codelist in self.bar_codelist_candidate_list:
            code_len = len(codelist)
            if code_len <= 3:
                continue
            check_code = codelist[code_len - 2]
            if not check_code.isdigit():
                continue

            # fill loss code with verification code
            if '*' in ''.join(codelist[1:code_len-2]):
                if display:
                    print('candidate with loss:', codelist)
                codelist_filled = self.bar_128_fill_loss(codelist[1:code_len - 2], check_code)
                if len(codelist_filled) > 0:
                    for sl in codelist_filled:
                        if sl not in self.result_code_possible:
                            self.result_code_possible.append([codelist[0]]+sl+[check_code]+[codelist[-1]])
                            self.result_code_valid = True
                            self.result_fill_loss += 1

            # verify and return result
            if '*' not in ''.join(codelist[1:code_len-2]):
                # if BarcodeReader128.get_codelist_check(codelist[1:code_len - 2], check_code, display):
                if self.checker.check_codelist(codelist=codelist, code_type=self.code_type)[0]:
                    self.result_code_valid = True
                    if codelist not in self.result_code_possible:
                        self.result_code_possible.append(codelist)

        if len(self.result_code_possible) > 0:
            self.result_codelist = self.get_maxscore_codelist_from_candidate(self.result_code_possible)
            self.result_code_valid = True
            self.result_code = ''.join([s for s in self.result_codelist[1:-2]
                                        if s not in ['CodeA', 'CodeB', 'CodeC']])

        return self.result_code_valid

    # deprecated now
    @staticmethod
    def get_codelist_checksum(code_list, display=False):
        chsum = 105
        codeb = 0
        chsum_list = [105]
        chsum0 = 0
        for i, s in enumerate(code_list):
            if s.isdigit() & (codeb == 1):
                chsum0 = (i+1)*(int(s)+16)
                codeb = 0
            elif s.isdigit():
                chsum0 = (i + 1) * int(s)
            elif s == 'CodeB':
                chsum0 = 600
                codeb = 1
            # print(s, chsum)
            chsum += chsum0
            chsum_list.append(chsum0)
        if display:
            print('item_check_value={0}, checksum={1}, check_result ={2}'.
                  format(chsum_list, chsum, chsum % 103))
        return chsum % 103

    @staticmethod
    def get_codelist_check(codelist, checksum, display=False):
        if type(checksum) == str:
            if checksum.isdigit():
                checksum = int(checksum)
            else:
                return False
        check_serial_sum_list = [105]
        codeb = 0
        for j, c in enumerate(codelist):
            # calculate check for CodeB/not
            if c == 'CodeB':
                check_serial_sum_list.append(100*(j+1))
                codeb = 1
            else:
                if codeb != 1:
                    if c.isdigit():
                        check_serial_sum_list.append(int(c) * (j+1))
                    else:
                        if display:
                            print('not digit code:{}!'.format(c))
                else:
                    if c.isdigit() & (len(c) == 1):
                        check_serial_sum_list.append((int(c)+16) * (j+1))
                    else:
                        check_serial_sum_list.append(-1)
                    codeb = 0
        check_valid = (sum(check_serial_sum_list) % 103) == checksum
        if display:
            print('check_sum: ', check_serial_sum_list, sum(check_serial_sum_list) % 103, checksum, check_valid)
        return check_valid

    def bar_128_fill_loss(self, code_list, check_code, display=False):
        result_list = []
        check_sum = int(check_code)
        loss_dict = {i: 0 for i, s in enumerate(code_list)
                     if (not s.isdigit()) & (s != 'CodeB')}
        loss_num = len(loss_dict)
        loss_keys = list(loss_dict.keys())
        if display:
            print('loss dict:', loss_dict)
        if loss_num == 0:
            if display:
                print('no loss checked')
            return code_list

        codebnum = ''.join(code_list).count('CodeB*')
        max_sum = 100 ** (loss_num-codebnum) * 10**codebnum
        cur_sum = 0
        while cur_sum < max_sum:
            for i in loss_keys:
                if code_list[i - 1] != 'CodeB':
                    if loss_dict[i] < 99:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                else:  # code_list[i - 1] == 'CodeB':
                    if loss_dict[i] < 10:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                loss_dict[i] = 0
            # check_code
            code_new = []
            for j, c in enumerate(code_list):
                if j not in loss_keys:
                    code_new.append(c)
                elif code_list[j-1] == 'CodeB':
                    code_new.append(str(loss_dict[j]))
                else:
                    if loss_dict[j] < 10:
                        code_new.append('0' + str(loss_dict[j]))
                    else:
                        code_new.append(str(loss_dict[j]))
            # if BarcodeReader128.get_codelist_check(code_new, check_sum):
            if self.checker.check_codelist(codelist=['Start']+code_new+['Stop'],
                                               code_type=self.code_type)[0]:
                result_list.append(code_new)
            cur_sum = cur_sum + 1
        return result_list  # codelist_list

    @staticmethod
    def bar_128_fill_loss2(code_list, display=False):
        """
        try to deal with 3 type code: codea,codeb,codec
        difficult is to deal with switch code, such as CodeB, FNC
        :param code_list: whole codelist, including startcode, endcode
        :param display: to display some commit messages
        :return: codelist with filled code
        """
        if len(code_list) < 4:
            if display:
                print('the length of codelist is too short')
            return code_list
        # fill ** with valid number
        # startcode = code_list[0]
        # endcode = code_list[-1]
        check_sum = int(code_list[-2])
        code_list = code_list[1:-2]
        result_list = []
        loss_dict = {i: 0 for i, s in enumerate(code_list)
                     if (not s.isdigit()) & (s.lower() not in 'codea,codeb,codec,fnc1,fnc4')}
        loss_num = len(loss_dict)
        loss_keys = list(loss_dict.keys())
        if display:
            print('loss dict:', loss_dict)
        if loss_num == 0:
            if display:
                print('no loss checked')
            return code_list

        codebnum = ''.join(code_list).count('CodeB*')
        max_sum = 100 ** (loss_num-codebnum) * 10**codebnum
        cur_sum = 0
        while cur_sum < max_sum:
            for i in loss_keys:
                if code_list[i - 1] != 'CodeB':
                    if loss_dict[i] < 99:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                else:  # code_list[i - 1] == 'CodeB':
                    if loss_dict[i] < 10:
                        loss_dict[i] = loss_dict[i] + 1
                        break
                loss_dict[i] = 0
            # check_code
            code_new = []
            for j, c in enumerate(code_list):
                if j not in loss_keys:
                    code_new.append(c)
                elif code_list[j-1] == 'CodeB':
                    code_new.append(str(loss_dict[j]))
                else:
                    if loss_dict[j] < 10:
                        code_new.append('0' + str(loss_dict[j]))
                    else:
                        code_new.append(str(loss_dict[j]))
            if BarcodeReader128.get_codelist_check(code_new, check_sum):
                result_list.append(code_new)
            cur_sum = cur_sum + 1

        # print('can not fill')
        return result_list  # code_list

    @staticmethod
    def generagte_barcode(code_str="1234567890", code_type='Code39'):
        from barcode.writer import ImageWriter
        from barcode import Code39, EAN8, EAN13, UPCA  # , upc
        from PIL import Image
        # from io import StringIO

        imagewriter = ImageWriter()
        # 保存到图片中
        # add_checksum : Boolean   Add the checksum to code or not (default: True)
        if code_type == 'Code39':
            ean = Code39(code_str, writer=imagewriter, add_checksum=False)
        elif code_type.upper() == 'EAN8':
            ean = EAN8(code_str, writer=imagewriter)  # , add_checksum=False)
        elif code_type.upper() == 'EAN13':
            ean = EAN13(code_str, writer=imagewriter)  # , add_checksum=False)
        # elif codetype.lower() == 'upc':
        #    ean = upc(codestr, writer=imagewriter)  # , add_checksum=False)
        elif code_type.upper() == 'UPCA':
            ean = UPCA(code_str, writer=imagewriter)  # , add_checksum=False)
        else:
            print('not suppoted codetype')
            return

        # 不需要写后缀，ImageWriter初始化方法中默认self.format = 'PNG'
        # print('保存到image2.png')
        ean.save('image2')
        img = Image.open('image2.png')
        # '展示image2.png'
        img.show()
        # img = plt.imread('image2.png')
        # plt.imshow(img)


class BarcodeTable:
    doc_string = \
       """
       code_type= 128a, 128b, 128c, 39, ...
       """

    def __init__(self, code_type):
        self.code_table = {}
        self.code_table_sno = {}
        self.code_table_sed = {}
        self.load_table(code_type)

    @abstractclassmethod
    def load_table(self, code_type='128c'):
        pass

    def get_code_from_bsnum(self, bsnum_str):
        if bsnum_str in self.code_table:
            return self.code_table[bsnum_str]
        else:
            return ''


class BarcodeTable128(BarcodeTable):

    def __init__(self, code_type):
        super().__init__(code_type)

    def load_table(self, code_type='128c'):
        if code_type.lower() not in ['128a', '128b', '128c']:
            print('invalid code type:{}'.format(code_type))
            return
        the_128table = self.__get_table_128_from_string().split('\n            ')
        for i, rs in enumerate(the_128table):
            s = rs.split('//')
            sk = s[0].replace(';', '')
            sa = s[1].strip()
            sb = s[2].strip()
            sc = s[3].strip()
            if i < 64:
                sb = sa
            sd = ''.join([str(int(sk[si]) + int(sk[si + 1])) for si in range(len(sk)-1)])
            self.code_table.update({sk: {'128a': sa, '128b': sb, '128c': sc}[code_type.lower()]})
            self.code_table_sno.update({{'128a': sa, '128b': sb, '128c': sc}[code_type.lower()]: i})
            self.code_table_sed.update({sd: {'128a': sa, '128b': sb, '128c': sc}[code_type.lower()]})

    @staticmethod
    def get_code_table_sed(code_table: dict):
        """
        get similar edge distance table
        dict = {similar_edge_distance: decoded_char}
        :param code_table:
        :return: similar edge distance table
        """
        if len(code_table) == 0:
            return {}
        code_table_se = {}
        for k in code_table:
            ss = ''
            for si in range(len(k)-1):
                ss += str(int(k[si])+int(k[si+1]))
            code_table_se[ss] = code_table[k]
        return code_table_se

    @staticmethod
    def __get_table_128_from_string():
        table_str = \
            '''2;1;2;2;2;2;//sp// =  //00
            2;2;2;1;2;2;// !// =   //01
            2;2;2;2;2;1;// "// =   //02
            1;2;1;2;2;3;// #// =   //03
            1;2;1;3;2;2;// $// =   //04
            1;3;1;2;2;2;// %// =   //05
            1;2;2;2;1;3;// &// =   //06
            1;2;2;3;1;2;//...// =  //07
            1;3;2;2;1;2;// (// =   //08
            2;2;1;2;1;3;// )// =   //09
            2;2;1;3;1;2;// *// =   //10
            2;3;1;2;1;2;// +// =   //11
            1;1;2;2;3;2;// ,// =   //12
            1;2;2;1;3;2;// -// =   //13
            1;2;2;2;3;1;// .// =   //14
            1;1;3;2;2;2;// / // =   //15
            1;2;3;1;2;2;// 0// =   //16
            1;2;3;2;2;1;// 1// =   //17
            2;2;3;2;1;1;// 2// =   //18
            2;2;1;1;3;2;// 3// =   //19
            2;2;1;2;3;1;// 4// =   //20
            2;1;3;2;1;2;// 5// =   //21
            2;2;3;1;1;2;// 6// =   //22
            3;1;2;1;3;1;// 7// =   //23
            3;1;1;2;2;2;// 8// =   //24
            3;2;1;1;2;2;// 9// =   //25
            3;2;1;2;2;1;// :// =   //26
            3;1;2;2;1;2;// ;// =   //27
            3;2;2;1;1;2;// <// =   //28
            3;2;2;2;1;1;// =// =   //29
            2;1;2;1;2;3;// >// =   //30
            2;1;2;3;2;1;// ?// =   //31
            2;3;2;1;2;1;// @// =   //32
            1;1;1;3;2;3;// A// =   //33
            1;3;1;1;2;3;// B// =   //34
            1;3;1;3;2;1;// C// =   //35
            1;1;2;3;1;3;// D// =   //36
            1;3;2;1;1;3;// E// =   //37
            1;3;2;3;1;1;// F// =   //38
            2;1;1;3;1;3;// G// =   //39
            2;3;1;1;1;3;// H// =   //40
            2;3;1;3;1;1;// I// =   //41
            1;1;2;1;3;3;// J// =   //42
            1;1;2;3;3;1;// K// =   //43
            1;3;2;1;3;1;// L// =   //44
            1;1;3;1;2;3;// M// =   //45
            1;1;3;3;2;1;// N// =   //46
            1;3;3;1;2;1;// O// =   //47
            3;1;3;1;2;1;// P// =   //48
            2;1;1;3;3;1;// Q// =   //49
            2;3;1;1;3;1;// R// =   //50
            2;1;3;1;1;3;// S// =   //51
            2;1;3;3;1;1;// T// =   //52
            2;1;3;1;3;1;// U// =   //53
            3;1;1;1;2;3;// V// =   //54
            3;1;1;3;2;1;// W// =   //55
            3;3;1;1;2;1;// X// =   //56
            3;1;2;1;1;3;// Y// =   //57
            3;1;2;3;1;1;// Z// =   //58
            3;3;2;1;1;1;// [// =   //59
            3;1;4;1;1;1;// \\// =  //60
            2;2;1;4;1;1;// ]// =   //61
            4;3;1;1;1;1;// ^// =   //62
            1;1;1;2;2;4;// _// =   //63
            1;1;1;4;2;2;// NUL//'//  64
            1;2;1;1;2;4;// SOH//a//  65
            1;2;1;4;2;1;// STX//b//  66
            1;4;1;1;2;2;// ETX//c//  67
            1;4;1;2;2;1;// EOT//d//  68
            1;1;2;2;1;4;// ENQ//e//  69
            1;1;2;4;1;2;// ACK//f//  70
            1;2;2;1;1;4;// BEL//g//  71
            1;2;2;4;1;1;// BS// h//  72
            1;4;2;1;1;2;// HT// i//  73
            1;4;2;2;1;1;// LF// j//  74
            2;4;1;2;1;1;// VT// k//  75
            2;2;1;1;1;4;// FF// l//  76
            4;1;3;1;1;1;// CR// m//  77
            2;4;1;1;1;2;// SO// n//  78
            1;3;4;1;1;1;// SI// o//  79
            1;1;1;2;4;2;// DLE//p//  80
            1;2;1;1;4;2;// DC1//q//  81
            1;2;1;2;4;1;// DC2//r//  82
            1;1;4;2;1;2;// DC3//s//  83
            1;2;4;1;1;2;// DC4//t//  84
            1;2;4;2;1;1;// NAK//u//  85
            4;1;1;2;1;2;// SYN//v//  86
            4;2;1;1;1;2;// ETB//w//  87
            4;2;1;2;1;1;// CAN//x//  88
            2;1;2;1;4;1;// EM// y//  89
            2;1;4;1;2;1;// SUB//z//  90
            4;1;2;1;2;1;// ESC//{//  91
            1;1;1;1;4;3;// FS// |//  92
            1;1;1;3;4;1;// GS// }//  93
            1;3;1;1;4;1;// RS// ~//  94
            1;1;4;1;1;3;// US// DEL//95
            1;1;4;3;1;1;//FNC3//FNC3//96
            4;1;1;1;1;3;//FNC2// FNC2//97
            4;1;1;3;1;1;//SHIFT//SHIFT//98
            1;1;3;1;4;1;//CodeC//CodeC//99
            1;1;4;1;3;1;//CodeB//FNC4//CodeB
            3;1;1;1;4;1;//FNC4//CodeA//CodeA
            4;1;1;1;3;1;//FNC1//FNC1//FNC1
            2;1;1;4;1;2;//StartA//StartA//StartA
            2;1;1;2;1;4;//StartB//StartB//StartB
            2;1;1;2;3;2;//StartC//StartC//StartC
            2;3;3;1;1;1;2;//Stop//Stop//Stop'''
        return table_str


# --- some useful functions in omrmodel or outside
class BarcodeUtil:

    def __init__(self):
        pass

    @staticmethod
    def show_image(fstr):
        if os.path.isfile(fstr):
            plt.imshow(mg.imread(fstr))
            plt.title(fstr)
            plt.show()
        else:
            print('file \"%s\" is not found!' % fstr)

    @staticmethod
    def find_file_from_pathfile(path_file):
        return path_file.replace('/', '\\').split('\\')[-1]

    @staticmethod
    def find_path_from_pathfile(path_file):
        ts = BarcodeUtil.find_file_from_pathfile(path_file)
        return path_file.replace(ts, '').replace('\\', '/')

    @staticmethod
    def glob_files_from_path(path, substr=''):
        if not os.path.isdir(path):
            return ['']
        file_list = []
        for f in glob.glob(path+'/*'):
            # print(f)
            if os.path.isfile(f):
                if len(substr) == 0:
                    file_list.append(f)
                elif substr in f:
                    file_list.append(f)
            if os.path.isdir(f):
                [file_list.append(s)
                 for s in BarcodeUtil.glob_files_from_path(f, substr)]
        return file_list

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

    @staticmethod
    def find_high_count_element(mylist: list):
        cn = Counter(mylist)
        if len(cn) > 0:
            return cn.most_common(1)[0][0]
        else:
            return 0

    @staticmethod
    def find_high_count_continue_element(mylist: list):
        if len(mylist) == 0:
            print('empty list')
            return -1
        countlist = [0 for _ in mylist]
        for i, e in enumerate(mylist):
            for ee in mylist[i:]:
                if ee == e:
                    countlist[i] += 1
                else:
                    break
        m = max(countlist)
        p = countlist.index(m)
        return mylist[p]

    @staticmethod
    def softmax(vector):
        sumvalue = sum([np.exp(v) for v in vector])
        return [np.exp(v)/sumvalue for v in vector]

    @staticmethod
    def image_slant_line(img, line_no, angle):
        if (line_no < 0) or (line_no >= img.shape[0]):
            print('invalid line_no:{}'.format(line_no))
            return
        img0 = []
        last = img[line_no, 0]
        for p in range(img.shape[1]):
            slant = line_no + int(np.tan(angle*3.14156/180)) * (p - img.shape[1])/2
            slant = int(slant)
            if 0 < slant < img.shape[0]:
                img0.append(img[slant, p])
                last = img[slant, p]
            else:
                img0.append(last)
        return img0

    @staticmethod
    def image_resize(img, reshape=(100, 200)):
        return cv2.resize(img, reshape)

    @staticmethod
    def image_amplify(img, ratio_row, ratio_col):
        return cv2.resize(img, (int(img.shape[1]*ratio_col), int(img.shape[0]*ratio_row)))

    @staticmethod
    def check_repeat_in_fields(df:pd.DataFrame, field_list):
        """
        check record repeated times on a field
        field_type must be str
        :param df: input dataframe
        :param field_list: fields to checked
        :return: dataframe to describe field repeated times count
        """
        result_df = pd.DataFrame({'field': [],
                                  'count': []})
        for f in field_list:
            if f in df.columns:
                gf = df.groupby(f)[f].count()
                gf = gf[gf > 1]
                rp = pd.DataFrame({'field': [fd for fd in gf.index],
                                   'count': [int(fc) for fc in gf.values]})
                result_df = result_df.append(rp)
        result_df.loc[:, 'count'] = result_df['count'].astype(int)
        return result_df


class BarCheckerFactory():

    def __init__(self):
        pass

    @staticmethod
    def create(code_type):
        if code_type == '128':
            return BarChecker128()
        else:
            print('invalid code_type:{}').format(code_type)
            return BarChecker()


class BarChecker(object):
    def __init__(self):
        self.code_type_list = []

    def check_codelist(self, codelist:list, display=False):
        print('no code type assigned, so check proc not set!')
        raise Exception
        return False


class BarChecker128(BarChecker):
    def __init__(self):
        super().__init__()
        self.code_type_list = ['128a', '128b', '128c']
        self.code_sno_dict = \
            {'128a': BarcodeTable128('128a').code_table_sno,
             '128b': BarcodeTable128('128b').code_table_sno,
             '128c': BarcodeTable128('128c').code_table_sno}
        self.code_esc_dict = \
            {'CodeA': self.code_sno_dict['128a'],
             'CodeB': self.code_sno_dict['128b'],
             'CodeC': self.code_sno_dict['128c']}

    def check_codelist(self, codelist, code_type=None, display=False):
        """
        calculate check result, include in list [check_validity, check_sum, code_checkvalue_list]
        :param codelist: list of barcode
        :param code_type: now, code_type is in ['128a', '128b', '128c']
        :return check_value_list: [True or False, checkvalue, [code_serial_no*index, ...]]
        """
        if type(codelist) != list:
            if display:
                print('error1: codelist is not list')
            return [False, -1, [-1]]
        if len(codelist) < 4:
            if display:
                print('error2: invalid length({}) codelist'.format(len(codelist)))
            return [False, -1, [-2 for _ in codelist]]
        if code_type not in self.code_sno_dict:
            if display:
                print('error3: invalid code type, not in {}'.format(self.code_sno_dict.keys()))
            return [False, -1, [-3 for _ in codelist]]
        if not codelist[-2].isdigit():
            if display:
                print('error4: check code is digit string')
            return [False, -1, [-4 for _ in codelist]]

        bt = self.code_sno_dict[code_type]
        ck_value_list = []
        ck_sum = 0
        for ci, cc in enumerate(codelist):
            if not isinstance(cc, str):
                ck_value_list.append('**')
            # check code in -2 and return
            if ci == len(codelist)-2:
                if ck_sum % 103 == int(codelist[-2]):
                    return [True, ck_sum % 103, ck_value_list]
                else:
                    return [False, ck_sum % 103, ck_value_list]
            if cc in bt:
                ck_value = bt.get(cc, -1) * (1 if ci == 0 else ci)
            elif ci > 0:
                # process escape code in mixed coding mode, eg: CodeB, n
                if codelist[ci-1] in self.code_esc_dict:
                    ck_value = self.code_esc_dict[codelist[ci - 1]].get(cc, -1) * ci
                else:
                    ck_value = -1
            else:  # ci==0: StartA,B,C
                ck_value = {'128a': 103, '128b': 104, '128c': 105}.get(code_type, -1)
            if ck_value >= 0:
                ck_sum += ck_value
                ck_value_list.append(ck_value)
            else:
                ck_value_list.append(cc)
            # print(ci, cc, cksum)
        return [False, ck_sum % 103, ck_value_list]


class BarDecoder:
    def __init__(self):
        self.code_type_list = None

    def decode(self, pwlist):
        pass


class BarDecoder128(BarDecoder):
    def __init__(self):
        super(BarDecoder128, self).__init__()
        self.code_type_list = ['128a', '128b', '128c']

    def decode(self,pwlist):
        pass
        return []
