# *_* utf-8 *-*


import numpy as np
import pandas as pd
import matplotlib.image as mg
import matplotlib.pyplot as plt
# from sklearn import svm
import time
import os


def readomr_task():
    omr = OmrRecog()
    # omr.omr_area_assign = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [22, 36]}
    # omr1: 'B84261310881005001_Omr01'          # horizon=37, vertical=14, validarea = [row:1-13, col:22-36]
    # omr2: '1a3119261913111631103_OMR01.jpg'   # horizon=31, vertical=6,
    #                                             validarea = [row:2-5;7-10;11-14;16-19;21-24;26-29, col:1-5]
    # omr3: '1a3119261913111631103_Oomr01.jpg'  # horizon=20, vertical=11,  validarea = [row:1-10, col:1-10]
    # omr1-->write to file:omr195_label, omr_block_(x,y)_xxx.jpg
    # fpath = r'C:\Users\wangxichang\students\ju\testdata\omr1'     # surface data
    # omr.set_omrformat([37, 14, 22, 36, 1, 13])
    # omr.savedatapath = r'C:\Users\wangxichang\students\ju\testdata\omr_result\omr-195'
    # omr2
    fpath = r'C:\Users\wangxichang\students\ju\testdata\omr2'     # surface data
    # fpath = r'f:\studies\juyunxia\omrimage1'      # 3-2 data
    # fpath = r'f:\studies\juyunxia\omrimage2'      # 3-2 data
    omr.set_omrformat([31, 6, 1, 30, 1, 5])
    omr.savedatapath = r'C:\Users\wangxichang\students\ju\testdata\omr_result\omr-150'
    # end
    flist = []
    for dirpath, dirnames, filenames in os.walk(fpath):
        for file in filenames:
            if ('.jpg' in file) & ('OMR' in file):
                flist.append(os.path.join(dirpath, file))
    readomr_result = None
    sttime = time.clock()
    runcount = 0
    for f in flist:
        print(runcount, '-->', f)
        omr.set_imgfilename(f)
        omr.run()
        if runcount == 0:
            readomr_result = omr.get_result_dataframe()
        else:
            readomr_result = readomr_result.append(omr.get_result_dataframe())
        # omr.save_result_omriamge()
        runcount += 1
    print(time.clock()-sttime, '\n', runcount)
    # for k in readomr_result:
    #    if len(readomr_result[k]) != 14:
    #        print(k, len(readomr_result[k]))
    # readomr_result
    return readomr_result


def test_one(fname, format, display=True):
    omr = OmrRecog()
    omr.imgfile = fname
    omr.set_omrformat(format)
    omr.display = display
    omr.run()
    return omr


def test(filename=''):
    omr = OmrRecog()
    if len(filename) == 0:
        # omrfile = 'omrtest3.jpg'
        # card: horizon=37, vertical=14, validarea = [row:1-13, col:22-36]
        omrfile = r'C:\Users\wangxichang\students\ju\testdata\omr1\B84261310881012002_Omr01.jpg'
        omr.omr_mark_area = {'mark_horizon_number': 37, 'mark_vertical_number': 14}
        omr.omr_valid_area = {'mark_horizon_number': [23, 36], 'mark_vertical_number': [1, 13]}
        # card:
        # omrfile = r'f:\studies\juyunxia\omrimage1\B84261310881012002_Omr01.jpg'  # 3-2 data
        # self.omr_mark_area = {'mark_horizon_number': 37, 'mark_vertical_number': 14}
        # self.omr_valid_area = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [23, 36]}
        # card
        # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_Oomr01.jpg'  # 3-2 data
        # omr.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        # omr.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        # card
        # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_OMR01.jpg'  # 3-2 data
    else:
        omrfile = filename
    omr.imgfile = omrfile
    omr.run()
    return omr


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class OmrRecog(object):
    def __init__(self):
        # input data and set parameters
        self.imgfile = ''
        self.img = None
        # valid area[rows cope, columns scope] for omr painting points
        self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        # system control parameters
        self.display = False        # display time, error messages in running process
        self.logwrite = False       # record processing messages in log file, finished later
        # inner parameter
        self.omr_threshold = 90
        self.omr_mean_threshold = 100
        self.check_vertical_window = 30
        self.check_horizon_window = 30
        self.check_step = 5
        # result data
        self.img_mean = -1
        self.img_std = -1
        self.omr_result = []
        self.xmap = None
        self.ymap = None
        self.omrxypos = [[], [], [], []]
        self.mark_omriamge = None
        self.recog_omriamge = None
        self.omrdict = {}
        self.omr_recog_data = {}
        self.omrsvm = None
        self.savedatapath = ''

    def test(self, filename=''):
        if len(filename) == 0:
            # card
            # omrfile = 'omrtest3.jpg'
            # omrfile = r'C:\Users\wangxichang\students\ju\testdata\omr1\B84261310881012002_Omr01.jpg'
            # omrfile = r'f:\studies\juyunxia\omrimage1\B84261310881012002_Omr01.jpg'  # 3-2 data
            # self.set_omrformat([20, 11, 1, 19, 1, 10])
            # self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
            # self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
            # card
            # self.omr_mark_area = {'mark_horizon_number': 37, 'mark_vertical_number': 14}
            # self.omr_valid_area = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [23, 36]}
            # card
            # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_Oomr01.jpg'  # 3-2 data
            # omrfile = r'c:\Users\wangxichang\students\ju\testdata\omr2\1a3119261913111631103_Oomr01.jpg'
            # self.set_omrformat([20, 11, 1, 19, 1, 10])
            # self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
            # self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
            # card
            # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_OMR01.jpg'  # 3-2 data
            omrfile = r'c:\Users\wangxichang\students\ju\testdata\omr2\1a3119261913111631103_OMR01.jpg'
            self.set_omrformat([31, 6, 1, 30, 1, 5])
        else:
            omrfile = filename
        self.imgfile = omrfile
        self.run()

    def run(self):
        # initiate some variables
        self.omrxypos = [[], [], [], []]
        # start running
        st = time.clock()
        self.get_img(self.imgfile)
        self.get_markblock()
        self.get_omrdict_xyimage()
        self.get_mark_omrimage()
        self.get_recog_data()
        self.get_recog_omrimage()
        print(f'consume {time.clock()-st}')

    def set_omrformat(self, cardform=(0, 0, 0, 0, 0, 0)):
        """
        :param
            format parameters, [mark_h_num, mark_v_num,
                               valid_h_start, valid_h_end,
                               valid_v_start, valid_v_end]
        :return
            False and pirnt messages if position to set is error
        """
        if (cardform[2] < 1) | (cardform[3] < cardform[2]) | (cardform[3] > cardform[0]):
            print(f'omr area setting error: mark start{cardform[2]}, end{cardform[3]}')
            return
        if (cardform[4] < 1) | (cardform[5] < cardform[4]) | (cardform[5] > cardform[1]):
            print(f'omr area setting error: mark start{cardform[2]}, end{cardform[3]}')
            return
        self.omr_mark_area['mark_horizon_number'] = cardform[0]
        self.omr_mark_area['mark_vertical_number'] = cardform[1]
        self.omr_valid_area['mark_horizon_number'] = [cardform[2], cardform[3]]
        self.omr_valid_area['mark_vertical_number'] = [cardform[4], cardform[5]]

    def set_imgfilename(self, fname):
        self.imgfile = fname

    def get_img(self, imfile):
        self.img = 255 - mg.imread(imfile)
        if len(self.img.shape) == 3:
            self.img = self.img.mean(axis=2)
        self.img_mean = self.img.mean()
        self.img_std = self.img.std()

    def get_result_dataframe(self):
        f = self.fun_findfilename(self.imgfile)
        return pd.DataFrame({'card': [f] * len(self.omr_recog_data['label']),
                             'coord': self.omr_recog_data['coord'],
                             'label': self.omr_recog_data['label'],
                             'saturation': self.omr_recog_data['saturation']})

    def save_result_omriamge(self):
        if self.savedatapath == '':
            print('to set save data path!')
            return
        if not os.path.exists(self.savedatapath):
            print(f'save data path "{self.savedatapath}" not exist!')
            return
        for coord in self.omrdict:
            f = self.savedatapath + '/omr_block_' + str(coord) + '_' + \
                self.fun_findfilename(self.imgfile)
            mg.imsave(f, self.omrdict[coord])

    # @staticmethod
    def fun_findfilename(self, pathfile):
        ts = pathfile
        ts.replace('/', '\\')
        p1 = ts.find('\\')
        if p1 > 0:
            ts = ts[::-1]; p1=ts.find('\\'); ts = ts[0 : p1]; ts = ts[::-1]
        return ts

    def get_markblock(self):
        # r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_pos(self.img, rowmark=True, step=3, window=30)
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom as img row bottom to remove noise
        rownum = self.img.shape[0]
        rownum = rownum - _step * _count
        r2, step, count = self.check_mark_pos(self.img[0:rownum, :],
                                              rowmark=False, step=5, window=30)
        if (len(r1[0]) > 0) | (len(r2[0]) > 0):
            self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])

    def check_mark_pos(self, img, rowmark, step, window):
        w = window
        vpol = self.img.shape[0] if rowmark else self.img.shape[1]
        mark_start_end_position = [[], []]
        count = 0
        while True:
            # no mark area found
            if vpol < w + step * count:
                print('check mark block in direction=', 'horizon'
                      if rowmark else 'vertical')
                print(f'no mark position found, checkline={vpol}, \
                     count={count}, step={step}, window={window}!')
                break
            imgmap = img[vpol - w - step * count:vpol - step * count, :].sum(axis=0) \
                         if rowmark else \
                         img[:, vpol - w - step * count:vpol - step * count].sum(axis=1)
            mark_start_end_position = self.check_mark_block(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    if self.display:
                        print('result:row={0},step={1},count={2}, imagescope={3}:{4}, marknum={5}'.
                              format(rowmark, step, count, vpol - w - step * count,
                              vpol - step * count, len(mark_start_end_position[0])))
                    return mark_start_end_position, step, count
            count += 1
        # print(f'no correct mark position solution found, row={rowmark}, step={step}, count={count}')
        return mark_start_end_position, step, count

    def check_mark_block(self, mapvec, rowmark):
        imgmapmean = mapvec.mean()
        mapvec[mapvec < imgmapmean] = 0
        mapvec[mapvec >= imgmapmean] = 1
        # smooth sharp peak and valley
        mapvec = self.check_mark_mapfun_smoothsharp(mapvec)
        if rowmark:
            self.xmap = mapvec
        else:
            self.ymap = mapvec
        # check mark positions
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
        # remove sharp peak -111-
        smooth_template = [-1, -1, 1, 1, 1, -1, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 3)[0] + 1] = 0
        rmap[np.where(ck == 3)[0] + 2] = 0
        rmap[np.where(ck == 3)[0] + 3] = 0
        # fill sharp valley
        smooth_template = [1, -2, 1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 1
        # remove sharp peak -1-
        smooth_template = [-1, 2, -1]
        ck = np.convolve(rmap, smooth_template, 'valid')
        rmap[np.where(ck == 2)[0] + 1] = 0
        return rmap

    def check_mark_result_evaluate(self, rowmark, poslist, step, count):
        poslen = len(poslist[0])
        # start position number is not same with end posistion number
        if poslen != len(poslist[1]):
            if self.display:
                print(f'startposnum{len(poslist[0])} != endposnum{len(poslist[1])}:', \
                      rowmark, step, count)
            return False
        # pos error: start pos less than end pos
        for pi in range(poslen):
            if poslist[0][pi] > poslist[1][pi]:
                if self.display:
                    print('start pos is less than end pos, in: ', rowmark, step, count)
                return False
        # width > 4 is considered valid mark block.
        tl = np.array([abs(x1-x2) for x1,x2 in zip(poslist[0],poslist[1])])
        validnum = len(tl[tl > 4])
        setnum = self.omr_mark_area['mark_horizon_number'] \
                if rowmark else \
                self.omr_mark_area['mark_vertical_number']
        if validnum != setnum:
            if self.display:
                print(f'valid mark num{validnum} != mark_set_num{setnum} in:', rowmark, step, count)
            return False
        if len(tl) != setnum:
            if self.display:
                print(f'checked mark num{len(t1)} != mark_set_num{setnum} in:', rowmark, step, count)
            return False
        # max width is too bigger than min width is a error result. 20%(3-5 pixels)
        maxwid = max(tl); minwid = min(tl); widratio = minwid/maxwid
        if widratio < 0.2:
            if self.display:
                print(f'maxwid={maxwid} / minwid={minwid} too small in:', rowmark, step, count)
            return False
        return True

    def get_omrdict_xyimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        if lencheck == 0:
            if self.display:
                print('no position vector created! so cannot create omrdict!')
            return
        # cut area for painting points
        for x in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                self.omrdict[(y, x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                                self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_block_saturability(self, blockmat):
        # visible pixel maybe 90 or bigger
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        comean = self.omr_threshold     #blockmat.mean()  # rowmean.mean()
        r1 = len(rowmean[rowmean > comean]) / len(rowmean)
        r2 = len(colmean[colmean > comean]) / len(colmean)
        st1 = round(max(r1, r2),4)
        # the number of big pixel
        bignum = len(blockmat[blockmat > self.omr_threshold])
        st2 = round(bignum / blockmat.size, 4)
        return st1, st2

    def get_mark_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        if lencheck == 0:
            if self.display:
                print('no position vector created! so cannot create omrdict!')
            return
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for col in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for row in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                omrimage[self.omrxypos[2][row]: self.omrxypos[3][row]+1,
                         self.omrxypos[0][col]: self.omrxypos[1][col]+1] = self.omrdict[(row, col)]
        self.mark_omriamge = omrimage

    def get_recog_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        if lencheck == 0:
            if self.display:
                print('no position vector created! so cannot create recog_omr_image!')
            return
        recogomr = np.zeros(self.img.shape)
        marknum = len(self.omr_recog_data['label'])
        for p in range(marknum):
            if self.omr_recog_data['label'][p] == 1:
                _x, _y = self.omr_recog_data['coord'][p]
                h1, h2 = [self.omr_valid_area['mark_horizon_number'][0] - 1,
                          self.omr_valid_area['mark_horizon_number'][1]]
                v1, v2 = [self.omr_valid_area['mark_vertical_number'][0] - 1,
                          self.omr_valid_area['mark_vertical_number'][1]]
                if (_x in range(v1, v2)) & (_y in range(h1, h2)):
                    recogomr[self.omrxypos[2][_x]: self.omrxypos[3][_x]+1,
                             self.omrxypos[0][_y]: self.omrxypos[1][_y]+1] \
                        = self.omrdict[(_x, _y)]
            p += 1
        self.recog_omriamge = recogomr
        # print(omr.omr_svmdata['label'][p],omr.omr_svmdata['coord'][p])

    # test use svm in sklearn
    def get_recog_data(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        if lencheck == 0:
            if self.display:
                print('no position vector! so cannot create recog_data[coord, \
                     data, label, saturation]!')
            return
        self.omr_recog_data = {'coord':[], 'data': [], 'label': [], 'saturation':[]}
        for j in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                painted_mean = self.omrdict[(i, j)].mean()
                # painted_std0 = self.omrdict[(i, j)].mean(axis=0).std()
                # painted_std1 = self.omrdict[(i, j)].mean(axis=1).std()
                # self.omr_recog_data['data'].append([painted_mean, painted_std0, painted_std1])
                self.omr_recog_data['data'].append(painted_mean)
                if painted_mean >= self.omr_mean_threshold:
                    # print(f'painted points: {(i, j)}')
                    self.omr_recog_data['label'].append(1)
                else:
                    self.omr_recog_data['label'].append(0)
                self.omr_recog_data['coord'].append((i, j))
                self.omr_recog_data['saturation'].append(
                    self.get_block_saturability(self.omrdict[(i, j)]))
        # clf = svm.LinearSVC()
        # clf.fit(self.omr_recog_data['data'], self.omr_recog_data['label'])
        # self.omrsvm = clf

    def get_recog_markcoord(self):
        if len(self.omr_recog_data['label']) == 0:
            print('no recog data!')
            return
        num = 0
        for coord, label in zip(self.omr_recog_data['coord'],
                                self.omr_recog_data['label']):
            if label == 1:
                num += 1
                print(num, coord)

    def plot_rawimage(self):
        plt.figure(1)
        plt.imshow(self.img)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap)

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap)

    def plot_mark_omrimage(self):
        if type(self.mark_omriamge) != np.ndarray:
            print('mark omr image is not created!')
            return
        plt.figure(4)
        plt.title('recognized - omr - region')
        plt.imshow(self.mark_omriamge)

    def plot_recog_omrimage(self):
        if type(self.recog_omriamge) != np.ndarray:
            print('mark omr image is not created!')
            return
        plt.figure(5)
        plt.title('recognized - omr - region')
        plt.imshow(self.recog_omriamge)

    def plot_omrblock_mean(self):
        plt.figure(6)
        data = self.omr_recog_data['data'].copy()
        data.sort()
        plt.plot([x for x in range(len(data))], data)
