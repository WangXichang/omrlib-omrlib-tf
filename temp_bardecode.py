
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import omrlib as ol
import form_test as ts

mid_vec = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


f8 = ts.form_8()
f8.file_list = f8.file_list[0:30]


def test():
    st = time.time()
    bar=ol.Barcoder()
    bar.set_image_clip(clip_bottom=600, clip_right=700)
    bar.set_image_files(f8.file_list)
    bar.get_barcode('128')
    print('used:', time.time()-st)
    return bar


def similar_merhod_decode128(bar_group, bar_wid):
    de = ''
    residue = 0
    for b in bar_group:
        if b % bar_wid == 0:
            residue = 0
        cc = b + residue
        if cc % bar_wid == 0:
            de = de + str(int(cc / bar_wid))
            residue = 0
        elif b < bar_wid:
            de = de + '1'
            residue = -bar_wid + 1
        else:
            de = de + str(int(cc/bar_wid))
            residue = cc - cc/bar_wid
    return de


class Barcoder:
    def __init__(self):
        self.codetype = 'code39'
        self.codelength = 0
        self.code_start_char_num = 1

        self.image_filename = None
        self.image = None
        self.gradient = None
        self.closed = None

        self.bar_image = None
        self.bar_image01 = None

        self.bar_widlist_dict = {}
        self.bar_result_dict = {}
        self.bar_result_code = ''
        self.bar_result_code_valid = False

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
        self.bar_code_128a = {}
        self.bar_code_128b = {}
        self.bar_code_128c = {}
        self.get_barcode_128table()

    def get_barcode_128table(self):
        self.bar_code_128dict = {}
        the_128table = self.get_128table().split('\n            ')
        # with open('barcode_128table.csv', 'r') as fp:
        for rs in the_128table:
            s = rs.split('//')
            sk = ''.join(s[0].split(';'))
            if s[1].strip() in ['StartA', 'StartB', 'StartC', 'Stop']:
                sa = sb = sc = s[1].strip()
            else:
                sa = s[1][1:6].strip()
                sb = s[1][7:12].strip()
                sc = s[1][13:].strip()
            self.bar_code_128a.update({sk: sa})
            self.bar_code_128b.update({sk: sb})
            self.bar_code_128c.update({sk: sc})

    def get_bar_image(self, filename='', clip_top=0, clip_bottom=0, clip_left=0, clip_right=0):
        if filename == '':
            if self.image_filename is not None:
                filename = self.image_filename
            else:
                print('not set image file to read!')
                return
        else:
            if not os.path.isfile(filename):
                print('filename error!')
                return

        # image = cv2.imread(args["image"])
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image[clip_top:image.shape[0] - clip_bottom,
                     clip_left:image.shape[1] - clip_right]

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction
        gradX = cv2.Sobel(self.image, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(self.image, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        self.gradient = cv2.convertScaleAbs(gradient)

        self.blurred = cv2.blur(gradient, (9, 9))
        # plt.imshow(self.blurred)

        (_, thresh) = cv2.threshold(self.blurred, 225, 255, cv2.THRESH_BINARY)
        # plt.imshow(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        self.closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.closed = cv2.erode(self.closed, None, iterations=4)
        self.closed = cv2.dilate(self.closed, None, iterations=4)
        # plt.imshow(self.closed)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        # (cnts, _) = cv2.findContours(self.closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img = self.closed
        img = cv2.normalize(self.closed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        (_, cnts, __) = cv2.findContours(img.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # cnt = cnts[0]
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.cv2.boxPoints(rect))
        # print(box)
        left, top, right, bottom = min(box[:, 0]) - 15, min(box[:, 1]) - 15, max(box[:, 0]) + 15, max(box[:, 1]) + 15

        # draw a bounding box arounded the detected barcode and display the image
        # cv2.drawContours(self.image, [box], -1, (0, 255, 0), 3)
        # cv2.imshow("Image", self.image)
        # cv2.waitKey(0)
        self.bar_image = self.image[top:bottom, left:right]

        img = 255 - self.bar_image.copy()
        th = img.mean() + 30
        img[img < th] = 0
        img[img >= th] = 1
        self.bar_image01 = img

        # get bar wid list
        for rowstep in range(-15, 15, 1):
            row = int(self.bar_image01.shape[0] * 1 / 2 + rowstep)
            mid_line = self.bar_image01[row]
            for j in range(len(mid_line)):
                if mid_line[j] == 1:
                    mid_line = mid_line[j:]
                    break
            for j in range(len(mid_line) - 1, 0, -1):
                if mid_line[j] == 1:
                    mid_line = mid_line[:j + 1]
                    break
            # vec = mid_line
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
            self.bar_widlist_dict[rowstep] = widlist

        # get result dict in mid_line[-15:15]
        self.bar_result_dict = dict()
        result_dict = dict()
        for j in range(-15, 15, 1):
            result = self.get_barcode(self.bar_widlist_dict[j])
            if len(result[0]) > 0:
                result_dict[j] = result[0]
            self.bar_result_dict[j] = result[0]

        # print(result_dict)
        # get code from result_dict
        valid_len = max([len(result_dict[x]) for x in result_dict if result_dict[x][0] != '**'])
        result_code = []
        for j in range(-15, 15, 1):
            if j not in result_dict:
                continue
            if len(result_dict[j]) == valid_len:
                if result_dict[j][0] != '**':
                    result0 = result_dict[j]
                else:
                    continue
            else:
                continue
            if len(result_code) == 0:
                result_code = result0
            else:
                for i, dc in enumerate(result0):
                    if (result_code[i] == '**') & (result0[i] != '**'):
                        result_code[i] = result0[i]
        check_sum = (105 + sum([(i+1)*int(x) for i, x in enumerate(result_code[1:-2])])) % 103
        if check_sum == int(result_code[-2]):
            self.bar_result_code_valid = True
        else:
            self.bar_result_code_valid =False
        self.bar_result_code = ''.join(result_code[1:-2])

    def get_barcode(self, widlist):
        if (len(widlist) - 1) % 6 > 0:
            # print('invalid widlist %s' % len(widlist))
            return '', '', []

        # seek code
        # wid_list = [widlist[i:i + 6] if i + 8 < len(widlist) else widlist[i:]
        #            for i in range(0, len(widlist), 6)
        #            if i < len(widlist)-6]
        # wid_list = wid_list[0:-1]
        wid_list = []
        for i in range(0, len(widlist), 6):
            if i < len(widlist)-8:
                wid_list.append(widlist[i:i+6])
            else:
                wid_list.append(widlist[i:])
                break

        codestr = ''
        for s in wid_list:
            sw = sum(s)
            si = ''
            for r in s:
                si = si + str(round(11 * r / sw))
            codestr = codestr + si
        # print(codestr)

        codetype = 'A' if codestr[0:6] == '211412' else \
                   'B' if codestr[0:6] == '211214' else \
                   'C'  # if codestr[0:6] == '211232' else '*'
        #if codetype == '*':
        #    print('code not 128')
        #    return None
        # else:
        # print(codetype)
        bar = Barcoder()
        decode_dict = bar.bar_code_128a if codetype == 'A' else \
            bar.bar_code_128b if codetype == 'B' else \
                bar.bar_code_128c
        result = []
        result2 = []
        for i in range(0, len(codestr), 6):
            dc = codestr[i:i + 6] if len(codestr) - i > 8 else codestr[i:]
            if dc in decode_dict:
                rdc = decode_dict[dc]
            else:
                rdc = '**'
            if rdc[0:4] == 'Stop':
                return result, result2, wid_list
            else:
                # result = result + rdc
                result.append(rdc)
            # result2 += (',' if i > 0 else '') + dc
            result2.append(dc)
            if i > len(codestr) - 8:
                break
        return result, result2, wid_list

    def show_bar_iamge(self):
        plt.figure('raw')
        plt.imshow(self.bar_image)

    def show_bar_iamge01(self):
        plt.figure('threshold 0-1')
        plt.imshow(self.bar_image01)

    def show_bar_width(self):
        print(self.bar_wid_list)

    def generagteBarCode(self, codestr="1234567890", codetype='Code39'):
        from barcode.writer import ImageWriter
        from barcode import Code39, EAN8, EAN13, upc, UPCA
        from PIL import Image
        from io import StringIO

        imagewriter = ImageWriter()
        # 保存到图片中
        # add_checksum : Boolean   Add the checksum to code or not (default: True)
        if codetype == 'Code39':
            ean = Code39(codestr, writer=imagewriter, add_checksum=False)
        elif codetype.upper() == 'EAN8':
            ean = EAN8(codestr, writer=imagewriter, add_checksum=False)
        elif codetype.upper() == 'EAN13':
            ean = EAN13(codestr, writer=imagewriter, add_checksum=False)
        elif codetype.lower() == 'upc':
            ean = upc(codestr, writer=imagewriter, add_checksum=False)
        elif codetype.upper() == 'UPCA':
            ean = UPCA(codestr, writer=imagewriter, add_checksum=False)
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

    def get_128table(self):
        table_str = '''2;1;2;2;2;2;// sp          00
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


