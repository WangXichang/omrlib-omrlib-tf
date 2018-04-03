# -*- utf8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mg
import numpy as np
import pandas as pd
import cv2, os, glob
# import os
# import glob
from collections import Counter


class Barcoder:
    def __init__(self):
        self.codetype_list = ['128', '39', 'ean8', 'ean13', 'upca']

        self.code_type = '128'
        self.code_num = -1
        self.code_unit_length = 2
        self.code_start_unit_num = 1
        self.code_check_unit_num = 1

        self.image_filenames = []
        self.image_clip_top = 0
        self.image_clip_bottom = 0
        self.image_clip_left = 0
        self.image_clip_right = 0
        self.image_threshold_shift = 65
        self.image_scan_scope = 12

        self.image_raw = None
        self.image_cliped = None
        self.image_gradient = None
        self.image_blurred = None
        self.image_closed = None
        self.image_bar = None
        self.image_bar01 = None

        self.bar_result_mlines_bslist_dict = {}
        self.bar_result_mlines_codelist_dict = {}
        self.bar_result_code_validclount_list = []
        self.bar_result_code_list = ''
        self.bar_result_code = ''
        self.bar_result_code_valid = False
        self.bar_result_dataframe = \
            pd.DataFrame({'fileno': [],
                          'filename': [],
                          'code': [],
                          'result': [],
                          'valid': []})

        self.bar_code_39 = {
            '0001101': 0, '0100111': 0, '1110010': 0,
            '0011001': 1, '0110011': 1, '1100110': 1,
            '0010011': 2, '0011011': 2, '1101100': 2,
            '0111101': 3, '0100001': 3, '1000010': 3,
            '0100011': 4, '0011101': 4, '1011100': 4,
            '0110001': 5, '0111001': 5, '1001110': 5,
            '0101111': 6, '0000101': 6, '1010000': 6,
            '0111011': 7, '0010001': 7, '1000100': 7,
            '0110111': 8, '0001001': 8, '1001000': 8,
            '0001011': 9, '0010111': 9, '1110100': 9,
        }
        self.table_128a = {}
        self.table_128b = {}
        self.table_128c = {}
        self._get_table_128()

    def set_image_files(self, file_list):
        self.image_filenames = file_list

    def set_image_clip(self, clip_top=0, clip_bottom=0, clip_left=0, clip_right=0):
        self.image_clip_top = clip_top
        self.image_clip_bottom = clip_bottom
        self.image_clip_left = clip_left
        self.image_clip_right = clip_right

    def get_barcode(self, code_type='128'):
        if self.code_num > 0:
            self.code_num = self.code_num
        else:
            print('code_num is not set')
            return
        if code_type == '128':
            for i, f in enumerate(self.image_filenames):
                self.get_barcode_128(f)
                print(i, Util.find_file_from_pathfile(f),
                      self.bar_result_code,
                      self.bar_result_code_list,
                      self.bar_result_code_valid)
                if i > 0:
                    self.bar_result_dataframe = \
                        self.bar_result_dataframe.append(
                            pd.DataFrame({
                                'fileno': [i],
                                'filename': [Util.find_file_from_pathfile(f)],
                                'code': [self.bar_result_code],
                                'result': [self.bar_result_code_list],
                                'valid': [self.bar_result_code_valid]
                                }, index=[i]))
                else:
                    self.bar_result_dataframe = \
                        pd.DataFrame({
                            'fileno': [i],
                            'filename': [Util.find_file_from_pathfile(f)],
                            'code': [self.bar_result_code],
                            'result': [self.bar_result_code_list],
                            'valid': [self.bar_result_code_valid]
                            }, index=[i])
                self.bar_result_dataframe.head()

    def get_barcode_128(self, filename):

        # init vars
        self.bar_result_mlines_bslist_dict = {}
        self.bar_result_mlines_codelist_dict = {}
        self.bar_result_code_validclount_list = []
        self.bar_result_code_list = []
        self.bar_result_code = ''
        self.bar_result_code_valid = False

        # preprocessing
        self._image_preprocessing(filename)

        # get 128code to result dict in mid_line[-scope:scope]
        # self.bar_result_dict = dict()
        mlines_code_dict = dict()
        for j in range(-self.image_scan_scope, self.image_scan_scope, 1):
            result = self._bar128_decode(self.bar_result_mlines_bslist_dict[j])
            if len(result) > 0:
                mlines_code_dict[j] = result
            self.bar_result_mlines_codelist_dict[j] = result
            # print(result)

        # get code from result_dict
        max_len = max([len(mlines_code_dict[x]) for x in mlines_code_dict])
        code_validcount_dict_list = [{} for _ in range(max_len)]
        for j in range(-self.image_scan_scope, self.image_scan_scope, 1):
            # not valid line or invalid bs list(no data items)
            if (j not in mlines_code_dict) or (len(mlines_code_dict[j]) < 4):
                continue
            for i, dc in enumerate(mlines_code_dict[j]):
                if dc.isdigit() or (dc in ['StartC', 'Stop', 'CodeB']):
                    if dc in code_validcount_dict_list[i]:
                        code_validcount_dict_list[i][dc] = code_validcount_dict_list[i][dc] + 1
                    else:
                        code_validcount_dict_list[i][dc] = 1
        self.bar_result_code_validclount_list = code_validcount_dict_list

        self.bar_result_code_list = self._bar_128_get_maxcount_code(code_validcount_dict_list)

        # get result_code
        valid_level = 3
        if (self.bar_result_code_list[0] != 'StartC') or \
                (self.bar_result_code_list[0] != 'Stop'):
            valid_level -= 1
        result_code = ''
        codeb = 0
        valid_list =[105]
        for j in range(1, max_len-1):
            # CodeB\code
            if codeb == 1:
                result_code += self.bar_result_code_list[j]
                if self.bar_result_code_list[j].isdigit() & \
                        (len(self.bar_result_code_list[j]) == 1):
                    valid_list.append((int(self.bar_result_code_list[j])+16)*j)
                else:
                    valid_list.append(-1)
                codeb += 1
                continue
            # check code
            if (codeb == 2) or (j == max_len-2):
                if self.bar_result_code_list[j].isdigit():
                    valid_list.append(int(self.bar_result_code_list[j]))
                else:
                    valid_list.append(-1)
                # check_code = self.bar_result_code_list[j]
                break
            if self.bar_result_code_list[j] != 'CodeB':
                result_code += self.bar_result_code_list[j]
                if self.bar_result_code_list[j].isdigit():
                    valid_list.append(int(self.bar_result_code_list[j])*j)
                else:
                    valid_list.append(-1)
            else:
                valid_list.append(100*j)
                codeb = 1
        self.bar_result_code = result_code

        # check valid
        # print(valid_list)
        self.bar_result_code_valid = \
            True if sum(valid_list[0:-1]) % 103 == valid_list[-1] else False

    def _bar_128_get_maxcount_code(self, code_validcount_list):
        result_code_list = []
        for dl in code_validcount_list:
            appended = False
            maxnum = max(dl.values()) if len(dl.values()) > 0 else 0
            for k in dl:
                if dl[k] == maxnum:
                    result_code_list.append(k)
                    appended = True
                    break
            if not appended:
                result_code_list.append('**')
        return result_code_list

    def _get_128_decode_old(self):
        docstrings = \
        '''
        # print(result_code_list)
        if (self.code_num > 0) & (max_len > self.code_num + 2):
            valid_len = self.code_num + 1 + 1  # start, code.., check, stop
        else:
            valid_len = -1

        result_code = ['' for _ in range(max_len)]
        for i, d in enumerate(result_code_count_dict):
            if len(d.values()) > 0:
                maxnum = max(d.values())
                for k in d:
                    if d[k] == maxnum:
                        if (self.code_num > 0) & (valid_len > 0):
                            if i in range(1, valid_len):
                                if not k.isdigit():
                                    result_code[i] = '**'
                                    break
                        result_code[i] = k
                        break
            else:
                if (self.code_num > 0) & (valid_len > 0):
                    if i <= valid_len:  # len(result_code_count_dict) - 1:
                        result_code[i] = '**'
                else:
                    result_code[i] = '**'
        self.bar_result_code_list = result_code
        # print(result_code)

        check_sum = (105 + sum(
            [(i + 1) * int(x) for i, x in enumerate(result_code[1:-2])
             if (len(x) > 0) & (x.isdigit())])) % 103
        self.bar_result_code_valid = False
        if result_code[-2].isdigit():  #(len(result_code[-2]) > 0) & (result_code[-2] != '**'):
            if check_sum == int(result_code[-2]):
                self.bar_result_code_valid = True

        self.bar_result_code = ''.join(result_code[1:1+self.code_num])
        if len(result_code) > self.code_num + 2:
            if result_code[self.code_num+1] != '**':
                checked = int(result_code[self.code_num+1])
                if '**' in result_code[1:1+self.code_num]:
                    self.bar_result_code = \
                        ''.join(self.bar_128_fill_loss(result_code[1:1+self.code_num], checked))
        # print(self.bar_result_code)
        '''

        pass
        return ''

    def _bar128_decode(self, barspace_pixels_list):
        if (len(barspace_pixels_list) - 1) % 6 > 0:
            # print('invalid widlist %s' % len(widlist))
            return ''  # , '', []

        # seek code
        wid_list = []
        for i in range(0, len(barspace_pixels_list), 6):
            if (i + 8) < len(barspace_pixels_list):
                wid_list.append(barspace_pixels_list[i:i + 6])
            else:
                wid_list.append(barspace_pixels_list[i:])
                break

        codestr = []
        for s in wid_list:
            sw = sum(s)
            si = ''
            bar_unit_len = 11 if len(s) == 6 else 13
            for r in s:
                si = si + str(round(bar_unit_len * r / sw))
            codestr.append(si)
        # print(codestr)

        codetype = 'A' if codestr[0:6] == '211412' else \
                   'B' if codestr[0:6] == '211214' else \
                   'C'  # if codestr[0:6] == '211232' else '*'

        # bar = Barcoder()
        decode_dict = \
            self.table_128a if codetype == 'A' else \
            self.table_128b if codetype == 'B' else \
            self.table_128c
        result = []
        codeb = 0
        for dc in codestr:  # range(0, len(codestr), 6):
            # dc = codestr[i:i + 6] if len(codestr) - i > 8 else codestr[i:]
            if dc in decode_dict:
                if codeb != 1:
                    rdc = decode_dict[dc]
                else:
                    rdc = self.table_128b[dc]
                    codeb = 0
                if (codetype == 'C') and (rdc == 'CodeB'):
                    codeb = 1
            else:
                rdc = '**'
            result.append(rdc)

        return result

    @staticmethod
    def bar_128_fill_loss(code_list, check_sum=0):
        loss_dict = {i: 0 for i, s in enumerate(code_list) if not s.isdigit()}
        loss_num = len(loss_dict)
        loss_keys = list(loss_dict.keys())
        if loss_num == 0:
            print('no loss checked')
            return code_list

        max_sum = 10 ** (loss_num+1)
        cur_sum = 0
        while cur_sum < max_sum:
            for i in loss_keys:
                if loss_dict[i] < 99:
                    loss_dict[i] = loss_dict[i] + 1
                    break
                else:
                    loss_dict[i] = 0
            # check_code
            code_new = [code_list[j] if j not in loss_keys else
                        (str(loss_dict[j]) if loss_dict[j] >= 10 else '0' + str(loss_dict[j]))
                        for j in range(len(code_list))]
            ch = (105 + sum([(h+1)*int(x) for h, x in enumerate(code_new)])) % 103
            # print(cur_sum, loss_dict, code_new, ch)
            if ch == check_sum:
                return code_new
                break
            cur_sum = cur_sum + 1

        return code_list

    def _image_preprocessing(self, filename):
        if (type(filename) != str) or (filename == ''):
            print('no image file given!')
            return
        else:
            if not os.path.isfile(filename):
                print('file %s not found!' % filename)
                return

        # read image,  from self.image_filenames
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_raw = image

        # clip image
        clip_top = self.image_clip_top
        clip_bottom = self.image_clip_bottom
        clip_left = self.image_clip_left
        clip_right = self.image_clip_right
        if (clip_top+clip_bottom<image.shape[0]) & \
                (clip_left+clip_right<image.shape[1]):
            self.image_cliped = image[clip_top:image.shape[0] - clip_bottom,
                                clip_left:image.shape[1] - clip_right]

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        gradX = cv2.Sobel(self.image_cliped, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(self.image_cliped, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        self.image_gradient = cv2.convertScaleAbs(gradient)

        self.image_blurred = cv2.blur(gradient, (9, 9))
        # plt.imshow(self.image_blurred)

        (_, thresh) = cv2.threshold(self.image_blurred, 225, 255, cv2.THRESH_BINARY)
        # plt.imshow(thresh)

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
        (_, contours, __) = cv2.findContours(img.copy(),
                                         mode=cv2.RETR_EXTERNAL,
                                         method=cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        # get bar image from box area
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.cv2.boxPoints(rect))
        left, top, right, bottom = min(box[:, 0]) - 15, \
                                   min(box[:, 1]) - 15, max(box[:, 0]) + 15, max(box[:, 1]) + 15
        self.image_bar = self.image_cliped[top:bottom, left:right]

        # binary image
        img = 255 - self.image_bar.copy()
        th = img.mean() + self.image_threshold_shift
        img[img < th] = 0
        img[img > 0] = 1
        self.image_bar01 = img

        # get bar wid list
        for rowstep in range(-self.image_scan_scope, self.image_scan_scope, 1):
            row = int(self.image_bar01.shape[0] * 1 / 2 + rowstep)
            mid_line = self.image_bar01[row]
            for j in range(len(mid_line)):
                if mid_line[j] == 1:
                    mid_line = mid_line[j:]
                    break
            for j in range(len(mid_line) - 1, 0, -1):
                if mid_line[j] == 1:
                    mid_line = mid_line[:j + 1]
                    break
            widlist = []
            last = mid_line[0]
            curwid = 1
            for j in range(1, len(mid_line)):
                cur = mid_line[j]
                if cur == last:
                    curwid += 1
                else:
                    widlist.append(curwid)
                    curwid = 1
                last = cur
            widlist.append(curwid)
            self.bar_result_mlines_bslist_dict[rowstep] = widlist

    def show_raw_iamge(self):
        plt.figure('raw image')
        plt.imshow(self.image_raw)

    def show_bar_iamge(self):
        plt.figure('gray bar image')
        plt.imshow(self.image_bar)

    def show_bar_iamge01(self):
        plt.figure('binary bar image')
        plt.imshow(self.image_bar01)

    def show_bar_bslist(self):
        bsdict = self.bar_result_mlines_bslist_dict
        for k in bsdict:
            print([bsdict[k][i:i+6] if i+7 < len(bsdict[k]) else bsdict[k][i:]
                   for i in range(0, len(bsdict[k]), 6)
                   if i+3 < len(bsdict[k])
                   ])

    def generagte_barcode(self, codestr="1234567890", codetype='Code39'):
        from barcode.writer import ImageWriter
        from barcode import Code39, EAN8, EAN13, UPCA  # , upc
        from PIL import Image
        # from io import StringIO

        imagewriter = ImageWriter()
        # 保存到图片中
        # add_checksum : Boolean   Add the checksum to code or not (default: True)
        if codetype == 'Code39':
            ean = Code39(codestr, writer=imagewriter, add_checksum=False)
        elif codetype.upper() == 'EAN8':
            ean = EAN8(codestr, writer=imagewriter)  # , add_checksum=False)
        elif codetype.upper() == 'EAN13':
            ean = EAN13(codestr, writer=imagewriter)  # , add_checksum=False)
        # elif codetype.lower() == 'upc':
        #    ean = upc(codestr, writer=imagewriter)  # , add_checksum=False)
        elif codetype.upper() == 'UPCA':
            ean = UPCA(codestr, writer=imagewriter)  # , add_checksum=False)
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

    def _get_table_128(self):
        self.bar_code_128dict = {}
        the_128table = self.__get_table_128_from_string().split('\n            ')
        for i, rs in enumerate(the_128table):
            s = rs.split('//')
            sk = s[0].replace(';', '')
            sp = s[1].strip()
            if sp in ['StartA', 'StartB', 'StartC', 'Stop']:
                sa = sb = sc = sp
            else:
                sa = s[1][1:6].strip()
                if i < 64:
                    sb = sa
                else:
                    sb = s[1][7:12].strip()
                sc = s[1][13:].strip()
            self.table_128a.update({sk: sa})
            self.table_128b.update({sk: sb})
            self.table_128c.update({sk: sc})

    @staticmethod
    def __get_table_128_from_string():
        table_str = \
            '''2;1;2;2;2;2;// sp          00
            2;2;2;1;2;2;// !           01
            2;2;2;2;2;1;// "           02
            1;2;1;2;2;3;// #           03
            1;2;1;3;2;2;// $           04
            1;3;1;2;2;2;// %           05
            1;2;2;2;1;3;// &           06
            1;2;2;3;1;2;// ...         07
            1;3;2;2;1;2;// (           08
            2;2;1;2;1;3;// )           09
            2;2;1;3;1;2;// *           10
            2;3;1;2;1;2;// +           11
            1;1;2;2;3;2;// ,           12
            1;2;2;1;3;2;// -           13
            1;2;2;2;3;1;// .           14
            1;1;3;2;2;2;// /           15
            1;2;3;1;2;2;// 0           16
            1;2;3;2;2;1;// 1           17
            2;2;3;2;1;1;// 2           18
            2;2;1;1;3;2;// 3           19
            2;2;1;2;3;1;// 4           20
            2;1;3;2;1;2;// 5           21
            2;2;3;1;1;2;// 6           22
            3;1;2;1;3;1;// 7           23
            3;1;1;2;2;2;// 8           24
            3;2;1;1;2;2;// 9           25
            3;2;1;2;2;1;// :           26
            3;1;2;2;1;2;// ;           27
            3;2;2;1;1;2;// <           28
            3;2;2;2;1;1;// =           29
            2;1;2;1;2;3;// >           30
            2;1;2;3;2;1;// ?           31
            2;3;2;1;2;1;// @           32
            1;1;1;3;2;3;// A           33
            1;3;1;1;2;3;// B           34
            1;3;1;3;2;1;// C           35
            1;1;2;3;1;3;// D           36
            1;3;2;1;1;3;// E           37
            1;3;2;3;1;1;// F           38
            2;1;1;3;1;3;// G           39
            2;3;1;1;1;3;// H           40
            2;3;1;3;1;1;// I           41
            1;1;2;1;3;3;// J           42
            1;1;2;3;3;1;// K           43
            1;3;2;1;3;1;// L           44
            1;1;3;1;2;3;// M           45
            1;1;3;3;2;1;// N           46
            1;3;3;1;2;1;// O           47
            3;1;3;1;2;1;// P           48
            2;1;1;3;3;1;// Q           49
            2;3;1;1;3;1;// R           50
            2;1;3;1;1;3;// S           51
            2;1;3;3;1;1;// T           52
            2;1;3;1;3;1;// U           53
            3;1;1;1;2;3;// V           54
            3;1;1;3;2;1;// W           55
            3;3;1;1;2;1;// X           56
            3;1;2;1;1;3;// Y           57
            3;1;2;3;1;1;// Z           58
            3;3;2;1;1;1;// [           59
            3;1;3;1;1;1;// \\           60
            2;2;1;4;1;1;// ]           61
            4;3;1;1;1;1;// ^           62
            1;1;1;2;2;4;// _           63
            1;1;1;4;2;2;// NUL   '     64
            1;2;1;1;2;4;// SOH   a     65
            1;2;1;4;2;1;// STX   b     66
            1;4;1;1;2;2;// ETX   c     67
            1;4;1;2;2;1;// EOT   d     68
            1;1;2;2;1;4;// ENQ   e     69
            1;1;2;4;1;2;// ACK   f     70
            1;2;2;1;1;4;// BEL   g     71
            1;2;2;4;1;1;// BS    h     72
            1;4;2;1;1;2;// HT    i     73
            1;4;2;2;1;1;// LF    j     74
            2;4;1;2;1;1;// VT    k     75
            2;2;1;1;1;4;// FF    l     76
            4;1;3;1;1;1;// CR    m     77
            2;4;1;1;1;2;// SO    n     78
            1;3;4;1;1;1;// SI    o     79
            1;1;1;2;4;2;// DLE   p     80
            1;2;1;1;4;2;// DC1   q     81
            1;2;1;2;4;1;// DC2   r     82
            1;1;4;2;1;2;// DC3   s     83
            1;2;4;1;1;2;// DC4   t     84
            1;2;4;2;1;1;// NAK   u     85
            4;1;1;2;1;2;// SYN   v     86
            4;2;1;1;1;2;// ETB   w     87
            4;2;1;2;1;1;// CAN   x     88
            2;1;2;1;3;1;// EM    y     89
            2;1;4;1;2;1;// SUB   z     90
            4;1;2;1;2;1;// ESC   {     91
            1;1;1;1;4;3;// FS    |     92
            1;1;1;3;4;1;// GS    }     93
            1;3;1;1;4;1;// RS    ~     94
            1;1;4;1;1;3;// US    DEL   95
            1;1;4;3;1;1;// FNC3  FNC3  96
            4;1;1;1;1;3;// FNC2  FNC2  97
            4;1;1;3;1;1;// SHIFT SHIFT 98
            1;1;3;1;4;1;// CodeC CodeC 99
            1;1;4;1;3;1;// CodeB FNC4  CodeB
            3;1;1;1;4;1;// FNC4  CodeA CodeA
            4;1;1;1;3;1;// FNC1  FNC1  FNC1
            2;1;1;4;1;2;//      StartA
            2;1;1;2;1;4;//      StartB
            2;1;1;2;3;2;//      StartC
            2;3;3;1;1;1;2;//     Stop'''
        return table_str

# --- some useful functions in omrmodel or outside
class Util:

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
        ts = Util.find_file_from_pathfile(path_file)
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
                 for s in Util.glob_files_from_path(f, substr)]
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
