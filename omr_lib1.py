# *_* utf-8 *-*

#from PIL import Image as im
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

imfilepath = r'c:\users\wangxichang\students\ju\testdata\omr2'

# img = im.open(r'c:\users\wangxichang\students\ju\testdata\omr2\1a3119261913111631103_OMR01.jpg')

# img2 = cv2.imread(r'c:\users\wangxichang\students\ju\testdata\omr2\1a3119261913111631103_OMR01.jpg')
class omrrecog():
    def __init__(self):
        self.img = None
        self.result = {'1':[[0,0], [1,1]]}
        self.imdf = None
        self.xmap = None
        self.ymap = None

    def test(self):
        self.get_img()
        self.convert_df()
        self.get_xpos()
        self.get_ypos()
        self.plotxmap()
        self.plotymap()

    def get_img(self):
        img2 = cv2.imread(imfilepath+r'\{}.jpg'.format('1a3119261913111631103_OMR01'))
        self.img =img2

    def convert_df(self):
        imdf = pd.DataFrame({str(x):self.img[:,x,1] for x in range(self.img.shape[1])})
        self.imdf = imdf.applymap(lambda x: 255-x)
        # self.imdf = imdf

    def get_xpos(self):
        u = pd.DataFrame(self.imdf.iloc[100:110,:].sum(axis=0))
        threshold = u.describe().loc['mean', 0]
        u = u.applymap(lambda x: 1 if x > threshold else 0)
        u['ind'] = u.index
        u['ind'] = u['ind'].apply(lambda x: int(x))
        u = u.sort_values('ind')
        #u.sort_values['ind']
        self.xmap = u

    def get_ypos(self):
        v = self.imdf.sum(axis=1)
        v = v.apply(lambda x: 1 if x > v.mean() else 0)
        self.ymap = v

    def show(self):
        # plt.figure(1)
        cv2.namedWindow('omr-image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('omr-image', self.img)

    def plotxmap(self):
        plt.figure(2)
        plt.plot(self.xmap.index, self.xmap[0])

    def plotymap(self):
        plt.figure(3)
        plt.plot(self.ymap.index, self.ymap.values)
