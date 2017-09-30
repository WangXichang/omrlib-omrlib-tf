# *_* utf-8 *-*


import numpy as np
import matplotlib.image as matimg
import matplotlib.pyplot as plt
from sklearn import svm
import time
import os


def readomr_task():
    omr = OmrRecog()
    #omr.omr_area_assign = {'row': [1, 13], 'col': [22, 36]}
    #fpath = r'C:\Users\wangxichang\students\ju\testdata\omr1'
    # fname = r'B84261310881005001_Omr01'
    omr.omr_set_area = {'row': [1, 5], 'col': [1, 29]}
    fpath = r'f:\studies\juyunxia\omrimage2'
    # '1a3119261913111631103_OMR01.jpg, _oomr01.jpg'
    flist = []
    for dirpath, dirnames, filenames in os.walk(fpath):
        for file in filenames:
            if '.jpg' in file:
                flist.append(os.path.join(dirpath, file))
    omr.omr_threshold = 50
    readomr_result = {}
    sttime = time.clock()
    runcount = 0
    for f in flist:
        print(f)
        omr.set_imgfilename(f)
        omr.run()
        readomr_result[f] = omr.omr_result
        runcount += 1
    print(time.clock()-sttime, '\n', runcount)
    for k in readomr_result:
        if len(readomr_result[k]) != 14:
            print(k, len(readomr_result[k]))
    return readomr_result


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class OmrRecog(object):
    def __init__(self):
        # input data and set parameters
        self.imgfile = ''
        self.img = None
        # valid area[rows cope, columns scope] for omr painting points
        self.omr_set_horizon_number = 30
        self.omr_set_vertical_number = 10
        self.omr_set_area = {'row': [1, 13], 'col': [22, 36]}
        # inner parameter
        self.omr_threshold = 60
        # result data
        self.omr_result = []
        self.xmap = None
        self.ymap = None
        self.omrxypos = []
        self.omr_width = -1
        self.omr_height = -1
        self.omrxnum = -1
        self.omrynum = -1
        self.omriamge = None
        self.omrdict = {}
        self.omr_svmdata = {}
        self.omrsvm = None

    def test(self, filename=''):
        if len(filename) == 0:
            omrfile = 'omrtest3.jpg'
        else:
            omrfile = filename
        self.omr_set_area = {'row': [1, 6], 'col': [1, 31]}
        # running
        st = time.clock()
        self.get_img(omrfile)
        self.get_markblock()
        self.get_omrdict_xyimage()
        # self.get_xyproj()
        # self.get_omr_xypos()
        # self.get_omrdict_xyimage()
        # self.get_omrimage()
        # self.get_svm()
        print(f'consume {time.clock()-st}')

    def run(self):
        self.get_img(self.imgfile)
        self.get_xyproj()
        self.get_omr_xypos()
        self.get_omr_result()

    def set_imgfilename(self, fname):
        self.imgfile = fname

    def get_img(self, imfile):
        self.img = 255 - matimg.imread(imfile)
        if len(self.img.shape) == 3:
            self.img = self.img.mean(axis=2)

    def get_markblock(self):
        # check horizonal mark blocks (columns number)
        r1 = self.check_mark_pos(rowmark=True, step=5, window=30, omr_block_number=self.omr_set_horizon_number)
        # check vertical mark blocks (rows number)
        r2 = self.check_mark_pos(rowmark=False, step=5, window=30, omr_block_number=self.omr_set_vertical_number)
        if (len(r1[0]) > 0) & (len(r2[0]) > 0):
            self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])

    def check_mark_pos(self, rowmark, step, window, omr_block_number):
        w = window
        vpol = self.img.shape[0] if rowmark else self.img.shape[1]
        mark_start_end_position = [[],[]]
        count = 0
        while True:
            if vpol < w + step * count:
                print('no goog pos found, vpol, count,step,window = ',vpol, count, step, window)
                break
            imgmap = self.img[vpol - w - step * count:vpol - step * count, :].sum(axis=0) \
                     if rowmark else \
                     self.img[:, vpol - w - step * count:vpol - step * count].sum(axis=1)
            imgmapmean = imgmap.mean()
            imgmap[imgmap < imgmapmean] = 0
            imgmap[imgmap >= imgmapmean] = 1
            imgmap = self.check_mark_mapfun_smoothsharp(imgmap)
            if rowmark:
                self.xmap = imgmap
                omr_num = self.omr_set_horizon_number
            else:
                self.ymap = imgmap
                omr_num = self.omr_set_vertical_number
            mark_start_end_position = self.check_mark_block(imgmap)
            if (len(mark_start_end_position[0]) == len(mark_start_end_position[1])) & \
                    (len(mark_start_end_position[0]) == omr_num):
                    # & self.evaluate_position_list(mark_start_end_position):
                    break
            count += 1
        print('result:', rowmark, step, count)
        return mark_start_end_position

    def check_mark_block(self, mapvec):
        mark_start_template = np.array([1, 1, 1, -1, -1])
        mark_end_template = np.array([-1, -1, 1, 1, 1])
        judg_value = 3
        r1 = np.convolve(mapvec, mark_start_template, 'valid')
        r2 = np.convolve(mapvec, mark_end_template, 'valid')
        # mark_position = np.where(r == 3)
        return np.where(r1 == judg_value)[0] + 2, np.where(r2 == judg_value)[0] + 2

    def check_mark_mapfun_smoothsharp(self, mapf):
        rmap = mapf
        # remove sharp peak -1-
        smooth_template = [-1, 2, -1]
        ck = np.convolve(mapf, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 0
        # remove sharp peak -11-
        smooth_template = [-1, 2, 2, -1]
        ck = np.convolve(mapf, smooth_template, 'valid')
        rmap[np.where(ck == 4)[0] + 1] = 0
        rmap[np.where(ck == 4)[0] + 2] = 0
        # fill sharp valley
        smooth_template = [1, -2, 1]
        ck = np.convolve(mapf, smooth_template, 'valid')[0] + 1
        rmap[np.where(ck == 2)[0] + 1] = 1
        return rmap

    def check_mark_pos_evaluate(self, poslist):
        if len(poslist[0]) > 50:
            return False
        tl = np.array([abs(x1-x2) for x1,x2 in zip(poslist[0],poslist[1])])
        rate = len(tl[tl > 4])
        if rate < min(self.omr_set_vertical_number, self.omr_set_horizon_number):
            return False
        else:
            return True

    def get_xyproj(self):
        rownum, colnum = self.img.shape
        omr_clip_row = 50     # columns mark points at bottom of page
        omr_clip_col = 80     # rows mark points at right of page
        # project to X-axis
        self.xmap = [self.img[rownum - omr_clip_row:rownum-1, x].sum() for x in range(colnum)]
        hmean = sum(self.xmap) / len(self.xmap)
        self.xmap = [1 if x > hmean else 0 for x in self.xmap]
        # project to Y-axis
        self.ymap = [self.img[y, colnum - omr_clip_col:colnum-1].sum() for y in range(rownum)]
        wmean = sum(self.ymap) / len(self.ymap)
        self.ymap = [1 if x > wmean else 0 for x in self.ymap]

    def get_omr_xypos(self):
        x_start = []
        x_end = []
        y_start = []
        y_end = []
        # model to detect start or end pos of mark points
        m = np.array([-1, -1, 1, 1, 1])
        m2 = np.array([1, 1, 1, -1, -1])
        for x in range(2, len(self.xmap)-2):
            if np.dot(self.xmap[x-2:x+3], m) == 3:
                x_start = x_start + [x]
            if np.dot(self.xmap[x - 2:x + 3], m2) == 3:
                x_end = x_end + [x]
        for y in range(2, len(self.ymap)-2):
            if np.dot(self.ymap[y - 2:y + 3], m) == 3:
                y_start = y_start + [y]
            if np.dot(self.ymap[y - 2:y + 3], m2) == 3:
                y_end = y_end + [y]
        # check x-start/x-end pairs
        if len(x_start) != len(x_end):
            print('check rows number from x-map fun error!')
            return False
        else:
            self.omrxnum = len(x_start)
        # check y-start/y-end pairs
        if len(y_start) != len(y_end):
            print('check columns number from x-map fun error!')
            return False
        else:
            self.omrynum = len(y_start)
        self.omrxypos = np.array([x_start, x_end, y_start, y_end])
        self.omr_width = [x2 - x1 for x1, x2 in zip(self.omrxypos[0], self.omrxypos[1])]
        self.omr_height = [x2 - x1 for x1, x2 in zip(self.omrxypos[2], self.omrxypos[3])]
        return True

    def get_omr_result(self):
        if self.omr_set_area['row'][1] > len(self.omrxypos[2]):
            print('row number is too big in omr_rwo_col!')
            return
        if self.omr_set_area['col'][1] > len(self.omrxypos[0]):
            print('col number is too big in omr_rwo_col!')
            return
        # cut area for painting points and set result to omr_result
        self.omr_result = []
        for y in range(self.omr_set_area['row'][0]-1, self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0]-1, self.omr_set_area['col'][1]):
                if self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                            self.omrxypos[0][x]:self.omrxypos[1][x]+1].mean() \
                        > self.omr_threshold:
                    self.omr_result.append((y+1, x+1))

    def get_omrdict_xyimage(self):
        if self.omr_set_area['row'][1] > len(self.omrxypos[2]):
            print('row number is too big in omr_rwo_col!')
            return
        if self.omr_set_area['col'][1] > len(self.omrxypos[0]):
            print('col number is too big in omr_rwo_col!')
            return
        # cut area for painting points
        for y in range(self.omr_set_area['row'][0]-1, self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0]-1, self.omr_set_area['col'][1]):
                self.omrdict[(y, x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                                self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_omrimage(self):
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for y in range(self.omr_set_area['row'][0], self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0], self.omr_set_area['col'][1]):
                omrimage[self.omrxypos[2][y]: self.omrxypos[3][y]+1,
                         self.omrxypos[0][x]: self.omrxypos[1][x]+1] = self.omrdict[(y, x)]
        self.omriamge = omrimage

    # test use svm in sklearn
    def get_svm(self):
        self.omr_svmdata = {'data': [], 'label': []}
        for i in range(self.omr_set_area['row'][0], self.omr_set_area['row'][1]):
            for j in range(self.omr_set_area['col'][0], self.omr_set_area['col'][1]):
                painted_mean = self.omrdict[(i, j)].mean()
                painted_std0 = self.omrdict[(i, j)].mean(axis=0).std()
                painted_std1 = self.omrdict[(i, j)].mean(axis=1).std()
                self.omr_svmdata['data'].append([painted_mean, painted_std0, painted_std1])
                # self.omr_svmdata['data'].append([painted_mean, painted_std0])
                if painted_mean >= 100:
                    # print(f'painted points: {(i, j)}')
                    self.omr_svmdata['label'].append(1)
                else:
                    self.omr_svmdata['label'].append(0)
        clf = svm.LinearSVC()
        clf.fit(self.omr_svmdata['data'], self.omr_svmdata['label'])
        self.omrsvm = clf

    def plot_rawimage(self):
        plt.figure(1)
        plt.imshow(self.img)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap)

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap)

    def plot_omrimage(self):
        plt.figure(4)
        plt.title('recognized - omr - region')
        plt.imshow(self.omriamge)

    def plot_mean_colstd_scatter(self):
        plt.figure(5)
        plt.scatter([x[0] for x in self.omr_svmdata['data']],
                    [x[1] for x in self.omr_svmdata['data']])

    def plot_mean_rowstd_scatter(self):
        plt.figure(6)
        plt.scatter([x[0] for x in self.omr_svmdata['data']],
                    [x[2] for x in self.omr_svmdata['data']])
