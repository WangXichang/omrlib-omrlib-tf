# *_* utf-8 *-*

#from PIL import Image as im
import cv2
import numpy as np
import pandas as pd
import matplotlib.image as matimg
import matplotlib.pyplot as plt


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class omrrecog():
    def __init__(self):
        self.img = None
        self.result = {'1':[[0,0], [1,1]]}
        self.imdf = None
        self.xmap = None
        self.ymap = None
        self.omrxy = []
        self.omrxwid = 0
        self.omrywid = 0
        self.omrxnum = 0
        self.omrynum = 0
        self.omriamge = None
        self.omrdict = {}

    def test(self):
        self.get_img()
        self.get_imageproject()
        self.get_omr_xyposition()
        self.get_omrdict()

    def get_img(self):
        # img = cv2.imread('omrtest0.jpg')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imdf = pd.DataFrame({str(x):self.img[:,x] for x in range(self.img.shape[1])})
        self.img = matimg.imread('omrtest0.jpg')
        imdf = pd.DataFrame(self.img)
        self.imdf = imdf.applymap(lambda x: 255-x)

    def get_imageproject(self):
        h, w = self.img.shape
        omr_clip_h = 50
        # project to X-axis
        u = pd.DataFrame(self.imdf.iloc[h - omr_clip_h:h, :].sum(axis=0))
        threshold = u.describe().loc['mean', 0]
        u = u.applymap(lambda x: 1 if x > threshold else 0)
        u['ind'] = u.index
        u['ind'] = u['ind'].apply(lambda x: int(x))
        u = u.sort_values('ind')
        self.xmap = u[[0]]
        # project to Y-axis
        v = pd.DataFrame(self.imdf.sum(axis=1))
        v[0] = v[0].apply(lambda x: 1 if x > v[0].mean() else 0)
        self.ymap = v

    def get_omr_xyposition(self):
        xlist = []
        xlist2 = []
        ylist = []
        ylist2 = []
        m = np.array([-1, 1, 1])
        m2 = np.array([1, 1, -1])
        for x in range(1, len(self.xmap)-1):
            if sum(self.xmap.loc[x-1:x+1, 0] * m) == 2:
                xlist = xlist +[x]
            if sum(self.xmap.loc[x-1:x+1, 0] * m2) == 2:
                xlist2 = xlist2 + [x]
        for y in range(1, len(self.ymap)-1):
            if sum(self.ymap.loc[y-1:y+1, 0] * m) == 2:
                ylist = ylist +[y]
            if sum(self.ymap.loc[y-1:y+1, 0] * m2) == 2:
                ylist2 = ylist2 +[y]
        if len(xlist) != len(xlist2):
            print('check x wave error!')
        else:
            self.omrxnum = len(xlist)
            self.omrxwid = round(sum([round(abs(x1 - x2)) for x1, x2 in zip(xlist, xlist2)]) / self.omrxnum)
        if len(ylist) != len(ylist2):
            print('check x wave error!')
        else:
            self.omrynum = len(ylist)
            self.omrxwid = round(sum([abs(x1 - x2) for x1, x2 in zip(ylist, ylist2)]) / self.omrynum)
        self.omrxy = [xlist, xlist2, ylist, ylist2]

    def get_omrdict(self):
        # omrimage = 255 - np.zeros([self.omrxnum, self.omrynum])
        omrimage = self.imdf.applymap(lambda x:0)
        for y in range(self.omrynum - 1):
            for x in range(self.omrxnum - 1):
                self.omrdict[(y,x)] = self.imdf.iloc[self.omrxy[2][y]:self.omrxy[3][y], \
                                                     self.omrxy[0][x]:self.omrxy[1][x]]
                omrimage.iloc[self.omrxy[2][y]:self.omrxy[3][y], self.omrxy[0][x]:self.omrxy[1][x]] = \
                    self.omrdict[(y,x)]
        self.omriamge = omrimage

    def show_rawimage(self):
        # cv2 mode
        # cv2.namedWindow('omr-image', cv2.WINDOW_AUTOSIZE)
        #  cv2.imshow('omr-image', self.img)
        # plt.figure(1)
        # matplotlib mode
        plt.imshow(self.img)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap.index, self.xmap[0])

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap.index, self.ymap.values)

    def plot_omrimage(self):
        plt.figure('recognized - omr - region')
        plt.imshow(self.omriamge)

