# *_* utf-8 *-*


import numpy as np
import pandas as pd
import matplotlib.image as matimg
import matplotlib.pyplot as plt
from sklearn import svm
import time


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class OmrRecog(object):
    def __init__(self):
        self.img = None
        # valid area[rows cope, columns scope] for omr painting points
        self.omr_row_col = {'row': [0, 12], 'col': [22, 36]}
        # self.imdf = None
        self.xmap = None
        self.ymap = None
        # self.omrxydict = []
        self.omrxwid = 0
        self.omrywid = 0
        self.omrxnum = 0
        self.omrynum = 0
        self.omriamge = None
        self.omrdict = {}
        self.omr_svmdata = {}
        self.omrsvm = None

    def test(self):
        omrfile = 'omrtest.jpg'
        st = time.clock()
        self.get_img(omrfile)
        self.get_xyproj()
        self.get_omr_xypos()
        self.get_omr_xyarea_dict()
        # self.get_omrimage()
        # self.get_svm()
        print(f'consume {time.clock()-st}')

    def get_img(self, imfile):
        self.img = 255 - matimg.imread(imfile)

    def get_xyproj(self):
        rowNum, colNum = self.img.shape
        omr_clip_row = 50     # columns mark points at bottom of page
        omr_clip_col = 80     # rows mark points at right of page
        # project to X-axis
        self.xmap = [self.img[rowNum - omr_clip_row:rowNum-1, x].sum() for x in range(colNum)]
        hmean = sum(self.xmap) / len(self.xmap)
        self.xmap = [1 if x > hmean else 0 for x in self.xmap]
        # project to Y-axis
        self.ymap = [self.img[y, colNum - omr_clip_col:colNum-1].sum() for y in range(rowNum)]
        wmean = sum(self.ymap) / len(self.ymap)
        self.ymap = [1 if x > wmean else 0 for x in self.ymap]

    def get_omr_xypos(self):
        x_start = []
        x_end = []
        y_start = []
        y_end = []
        # model to detect start or end pos of mark points
        m = np.array([-1, 1, 1])
        m2 = np.array([1, 1, -1])
        for x in range(1, len(self.xmap)-1):
            if np.dot(self.xmap[x-1:x+2], m) == 2:
                x_start = x_start + [x]
            if np.dot(self.xmap[x - 1:x + 2], m2) == 2:
                x_end = x_end + [x]
        for y in range(1, len(self.ymap)-1):
            if np.dot(self.ymap[y-1:y+2], m) == 2:
                y_start = y_start + [y]
            if np.dot(self.ymap[y-1:y+2], m2) == 2:
                y_end = y_end + [y]
        # check x-start/x-end pairs
        if len(x_start) != len(x_end):
            print('check x wave error!')
        else:
            self.omrxnum = len(x_start)
        # check y-start/y-end pairs
        if len(y_start) != len(y_end):
            print('check x wave error!')
        else:
            self.omrynum = len(y_start)
        # return x-pos, y-pos in list
        # self.omrxydict = {'x_start':x_start, 'x_end':x_end, 'y_start':y_start, 'y_end':y_end}
        self.omrxypos = np.array([x_start, x_end, y_start, y_end])

    def get_omr_xyarea_dict(self):
        # cut area for painting points
        for y in range(self.omr_row_col['row'][0], self.omr_row_col['row'][1]):
            for x in range(self.omr_row_col['col'][0], self.omr_row_col['col'][1]):
                self.omrdict[(y,x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                               self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_omrimage(self):
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for y in range(self.omr_row_col['row'][0], self.omr_row_col['row'][1]):
            for x in range(self.omr_row_col['col'][0], self.omr_row_col['col'][1]):
                omrimage[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                         self.omrxypos[0][x]:self.omrxypos[1][x]+1] = self.omrdict[(y,x)]
        self.omriamge = omrimage

    # test use svm in sklearn
    def get_svm(self):
        self.omr_svmdata = {'data':[], 'label':[]}
        for i in range(self.omr_row_col['row'][0],self.omr_row_col['row'][1]):
            for j in range(self.omr_row_col['col'][0],self.omr_row_col['col'][1]):
                painted_mean = self.omrdict[(i,j)].mean().mean()
                painted_std0 = self.omrdict[(i,j)].sum(axis=0).std()
                # painted_std1 = self.omrdict[(i,j)].sum(axis=1).std()
                # self.omr_svmdata['data'].append([painted_mean, painted_std0, painted_std1])
                self.omr_svmdata['data'].append([painted_mean, painted_std0])
                if painted_mean >= 100:
                    # print(f'painted points: {(i, j)}')
                    self.omr_svmdata['label'].append(1)
                else:
                    self.omr_svmdata['label'].append(0)
        clf = svm.LinearSVC()
        clf.fit(self.omr_svmdata['data'], self.omr_svmdata['label'])
        self.omrsvm = clf

    def show_rawimage(self):
        plt.imshow(self.img)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap)

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap)

    def plot_omrimage(self):
        plt.figure('recognized - omr - region')
        plt.imshow(self.omriamge)
