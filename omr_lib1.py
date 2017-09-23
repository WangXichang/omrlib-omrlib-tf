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

    def get_img(self):
        img2 = cv2.imread(imfilepath+r'\{}.jpg'.format('1a3119261913111631103_OMR01'))
        self.img =img2

    def convert_df(self):
        imdf = pd.DataFrame({str(x):self.img[:,x,1] for x in range(self.img.shape[1])})
        imdf.applymap(lambda x: 255-x)
        self.imdf = imdf

    def get_xpos(self):
        u = self.imdf.sum(axis=0)
        #u['ind'] = u.index
        #u['ind'] = u['ind'].apply(lambda x: int(x))
        #u.sort_values['ind']
        plt.plot(u.index, u.values)

    def get_ypos(self):
        v = self.imdf.sum(axis=1)
        v = v.apply(lambda x: 1 if x > v.mean() else 0)
        plt.plot(v)
