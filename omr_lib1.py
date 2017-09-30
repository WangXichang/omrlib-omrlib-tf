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
        self.omr_set_horizon_number = 20
        self.omr_set_vertical_number = 11
        self.omr_set_area = {'row': [1, 10], 'col': [1, 20]}
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
        self.omr_set_horizon_number = 20
        self.omr_set_vertical_number = 11
        self.omr_set_area = {'row': [1, 10], 'col': [1, 20]}
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
        r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_pos(self.img, rowmark=True, step=5, window=30)
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom line to remove noise
        rownum, _ = self.img.shape
        if len(r1[0]) > 0:
            r2, step, count = self.check_mark_pos(self.img[0:rownum - _step * _count, :], \
                                                  rowmark=False, step=5, window=30)
        self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])

    def check_mark_pos(self, img, rowmark, step, window):
        w = window
        vpol = self.img.shape[0] if rowmark else self.img.shape[1]
        mark_start_end_position = [[],[]]
        count = 0
        while True:
            if vpol < w + step * count:
                print('check mark block in direction=', 'horizon' if rowmark else 'vertical')
                print('no goog pos found, vpol, count,step,window = ',vpol, count, step, window)
                break
            imgmap2 = self.check_mark_mapfun_smoothsharp(
                        img[vpol - w - step * count:vpol - step * count, :].sum(axis=0)
                        if rowmark else
                        img[:, vpol - w - step * count:vpol - step * count].sum(axis=1))
            if rowmark:
                self.xmap = imgmap2
                omr_num = self.omr_set_horizon_number
            else:
                self.ymap = imgmap2
                omr_num = self.omr_set_vertical_number
            mark_start_end_position = self.check_mark_block(imgmap2)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position):
                    # & self.evaluate_position_list(mark_start_end_position):
                    print('result:row={0},step={1},count={2}, imagescope={3}:{4}, marknum={5}'. \
                          format(rowmark, step, count, vpol - w - step * count, vpol - step * count,
                                 len(mark_start_end_position[0])))
                    return mark_start_end_position, step, count
            count += 1
        print(f'no correct mark position solution found, row={rowmark}, step={step}, count={count}')
        return mark_start_end_position, step, count

    def check_mark_block(self, mapvec):
        imgmapmean = mapvec.mean()
        mapvec[mapvec < imgmapmean] = 0
        mapvec[mapvec >= imgmapmean] = 1
        mark_start_template = np.array([1, 1, 1, -1, -1])
        mark_end_template = np.array([-1, -1, 1, 1, 1])
        judg_value = 3
        r1 = np.convolve(mapvec, mark_start_template, 'valid')
        r2 = np.convolve(mapvec, mark_end_template, 'valid')
        # mark_position = np.where(r == 3)
        return np.where(r1 == judg_value)[0] + 2, np.where(r2 == judg_value)[0] + 2

    def check_mark_mapfun_smoothsharp(self, mapf):
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
        # fill sharp valley
        smooth_template = [1, -2, 1]
        ck = np.convolve(rmap, smooth_template, 'valid')[0]
        rmap[np.where(ck == 2)[0] + 1] = 1
        return rmap

    def check_mark_result_evaluate(self, rowmark, poslist):
        if len(poslist[0]) != len(poslist[1]):
            # print('start poslist is not same len as end poslist!')
            return False
        tl = np.array([abs(x1-x2) for x1,x2 in zip(poslist[0],poslist[1])])
        validnum = len(tl[tl > 4])
        setnum = self.omr_set_horizon_number if rowmark else self.omr_set_vertical_number
        if validnum != setnum:
            print(poslist)
            print(f'mark block row={rowmark},valid number={validnum}, setnumber={setnum}')
            return False
        else:
            return True

    def get_omr_result(self):
        # cut area for painting points and set result to omr_result
        self.omr_result = []
        for y in range(self.omr_set_area['row'][0]-1, self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0]-1, self.omr_set_area['col'][1]):
                if self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                            self.omrxypos[0][x]:self.omrxypos[1][x]+1].mean() \
                        > self.omr_threshold:
                    self.omr_result.append((y+1, x+1))

    def get_omrdict_xyimage(self):
        # cut area for painting points
        for y in range(self.omr_set_area['row'][0]-1, self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0]-1, self.omr_set_area['col'][1]):
                self.omrdict[(y, x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                                self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_omrimage(self):
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for y in range(self.omr_set_area['row'][0]-1, self.omr_set_area['row'][1]):
            for x in range(self.omr_set_area['col'][0]-1, self.omr_set_area['col'][1]):
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
