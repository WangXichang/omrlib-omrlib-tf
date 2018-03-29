
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

mid_vec = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
       0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



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
        self.image_filename = None
        self.image = None
        self.gradient = None
        self.closed = None
        self.bar_image = None
        self.bar_image01 = None
        self.bar_widlist_dict = {}

        self.bar_decode_dict = {
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
        with open('barcode_128table.csv', 'r') as fp:
            for rs in fp.readlines():
                rs = rs.replace('\n', '')
                # print(rs.split('//'))
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
        th = img.mean()
        img[img < th] = 0
        img[img >= th] = 1
        self.bar_image01 = img

        # get bar wid list
        for rowstep in range(-10, 10, 1):
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

    def get_barcode(self, widlist):

        if (len(widlist) - 1) % 6 > 0:
            print('invalid widlist %s' % len(widlist))
            return None

        # seek code
        wid_list = [widlist[i:i + 6] if i + 8 < len(widlist) else widlist[i:] for i in range(0, len(widlist), 6)]
        wid_list = wid_list[0:-1]

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
        result = ''
        result2 = ''
        for i in range(6, len(codestr), 6):
            dc = codestr[i:i + 6] if len(codestr) - i > 10 else codestr[i:]
            if dc in decode_dict:
                rdc = decode_dict[dc]
            else:
                rdc = '**'
            if rdc == 'Stop':
                return result, result2, widlist
            else:
                result = result + rdc
            result2 += (',' if i > 6 else '') + dc
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

        '''
        # 写入stringio流中
        bar_io = StringIO()
        ean = Code39(codestr, writer=imagewriter, add_checksum=False)
        ean.write(bar_io)
        bar_io = StringIO(bar_io.getvalue())
        img1 = Image.open(bar_io)
        # 在stringIO中以图片方式打开'
        img1.show()
        '''

