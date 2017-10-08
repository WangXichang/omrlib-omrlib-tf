# *_* utf-8 *-*


import numpy as np
import pandas as pd
import matplotlib.image as mg
import matplotlib.pyplot as plt
# from sklearn import svm
from sklearn.cluster import KMeans
import time
import os


def readomr_task(cardno):
    fpath, cardformat, flist = card(cardno)
    omr = OmrModel()
    omr.set_format(cardformat)
    #
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
    # fpath = r'C:\Users\wangxichang\students\ju\testdata\omr2'     # surface data
    # fpath = r'f:\studies\juyunxia\omrimage1'      # 3-2 data
    # fpath = r'f:\studies\juyunxia\omrimage2'      # 3-2 data
    # omr.set_format([31, 6, 1, 30, 1, 5])
    omr.savedatapath = r'C:\Users\wangxichang\students\ju\testdata\omr_result\omr-150'
    # end
    # flist = []
    # for dirpath, dirnames, filenames in os.walk(fpath):
    #    for file in filenames:
    #        if ('.jpg' in file) & ('OMR' in file):
    #            flist.append(os.path.join(dirpath, file))
    readomr_result = None
    sttime = time.clock()
    runcount = 0
    for f in flist:
        print(runcount, '-->', f)
        omr.set_img(f)
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
    rg3=readomr_result[readomr_result.label2==1]\
        [['card','coord','label','satu']].groupby('card').count()
    return readomr_result, rg3


def test_one(fname, cardformat, display=True):
    omr = OmrModel()
    omr.imgfile = fname
    omr.set_format(cardformat)
    omr.display = display
    omr.run()
    r = omr.get_result_dataframe()
    print(r[r.label2==1][['coord','label','satu']])
    return omr, r


def test(filename=''):
    omr = OmrModel()
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


def card(no):
    filterfile = ['.jpg']
    fpath = ''
    cardformat = []
    if no == 1:
        fpath = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\'
        cardformat = [37, 14, 23, 36, 1, 13]
    elif no == 2:  # OMR..jpg
        filterfile = filterfile + ['OMR']
        fpath = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\'
        cardformat = [31, 6, 1, 30, 1, 5]
    elif no == 3:  # Oomr..jpg
        filterfile = filterfile + ['Oomr']
        fpath = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\'
        cardformat = [20, 11, 1, 19, 1, 10]
    flist = []
    for dirpath, dirnames, filenames in os.walk(fpath):
        for file in filenames:
            b = True
            for ss in filterfile:
                b = b & (ss in file)
            if b:
                flist.append(os.path.join(dirpath, file))
    return fpath, cardformat, flist


# read omr card image and recognized the omr painting area(points)
# further give detect function to judge whether the area is painted
class OmrModel(object):
    """
    processing omr image class
    set data: set_img, set_format
        imgfile: omr image file name string
        format: [mark horizon nunber, mark vertical number, valid_h_start,end, valid_v_start,end]
        savedatapath: string, path to save omr block images file in save_result_omriamge()
    set para:
        display: bool, display meassage in runtime
        logwrite: bool, write to logger, not implemented yet
    result:
        omrdict: dict, (x,y):omrblcok image matrix ndarray
        omr_recog_data: dict
        omrxypos: list, [[x-start-pos,], [x-end-pos,],[y-start-pos,], [y-end-pos,]]
        xmap: list
        ymap: list
        mark_omrimage:
        recog_omrimage:_
    inner para:
        omr_threshold:int, gray level to judge painted block
        check_
    """
    def __init__(self):
        # input data and set parameters
        self.imgfile = ''
        self.img = None
        self.savedatapath = ''
        # valid area[rows cope, columns scope] for omr painting points
        self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        # system control parameters
        self.display = False        # display time, error messages in running process
        self.logwrite = False       # record processing messages in log file, finished later
        # inner parameter
        self.omr_threshold = 75
        self.check_vertical_window = 30
        self.check_horizon_window = 20
        self.check_step = 10
        # result data
        # self.img_mean = -1
        # self.img_std = -1
        self.xmap = None
        self.ymap = None
        self.omrxypos = [[], [], [], []]
        self.mark_omriamge = None
        self.recog_omriamge = None
        self.omrdict = {}
        self.omr_recog_data = {}
        self.omrsvm = None

    def run(self):
        # initiate some variables
        self.omrxypos = [[], [], [], []]
        # start running
        st = time.clock()
        self.get_img(self.imgfile)
        self.get_markblock()
        self.get_omrdict_xyimage()
        self.get_recog_data()
        # self.get_mark_omrimage()
        # self.get_recog_omrimage()
        print(f'consume {time.clock()-st}')

    def set_format(self, cardform=(0, 0, 0, 0, 0, 0)):
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

    def set_img(self, fname):
        self.imgfile = fname

    def get_img(self, imfile):
        self.img = 255 - mg.imread(imfile)
        if len(self.img.shape) == 3:
            self.img = self.img.mean(axis=2)
        # self.img_mean = self.img.mean()
        # self.img_std = self.img.std()

    def save_result_omriamge(self):
        if self.savedatapath == '':
            print('to set save data path!')
            return
        if not os.path.exists(self.savedatapath):
            print(f'save data path "{self.savedatapath}" not exist!')
            return
        for coord in self.omrdict:
            f = self.savedatapath + '/omr_block_' + str(coord) + '_' + \
                self.fun_findfile(self.imgfile)
            mg.imsave(f, self.omrdict[coord])

    def get_markblock(self):
        # r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_pos(self.img,
                                                rowmark=True,
                                                step=self.check_step,
                                                window=self.check_horizon_window)
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom as img row bottom to remove noise
        rownum = self.img.shape[0]
        rownum = rownum - _step * _count
        r2, step, count = self.check_mark_pos(self.img[0:rownum, :],
                                              rowmark=False,
                                              step=self.check_step,
                                              window=self.check_vertical_window)
        if (len(r1[0]) > 0) | (len(r2[0]) > 0):
            self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])
            # adjust peak width
            self.check_mark_adjustpeak()

    def check_mark_pos(self, img, rowmark, step, window):
        w = window
        maxlen = self.img.shape[0] if rowmark else self.img.shape[1]
        mark_start_end_position = [[], []]
        count = 0
        while True:
            # no mark area found
            if maxlen < w + step * count:
                if self.display:
                    direction = 'horizon' if rowmark else 'vertical'
                    print(f'marks not found in direction={direction}, \
                         checkline={maxlen- w - step*(count-1)}:{maxlen - w + step*count}, \
                         count={count}, step={step}, window={window}!')
                break
            imgmap = img[maxlen - w - step * count:maxlen - step * count, :].sum(axis=0) \
                         if rowmark else \
                         img[:, maxlen - w - step * count:maxlen - step * count].sum(axis=1)
            mark_start_end_position = self.check_mark_block(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    if self.display:
                        direction = 'horizon' if rowmark else 'vertical'
                        print('markcheck:dir={0},step={1},count={2},imgzone={3}:{4}, mark={5}'.
                              format(direction, step, count, maxlen - w - step * count,
                              maxlen - step * count, len(mark_start_end_position[0])))
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

    def check_mark_adjustpeak(self):
        # return
        peaknum = len(self.omrxypos[0])
        pw = [self.omrxypos[1][i] - self.omrxypos[0][i] for i in range(peaknum)]
        vw = [self.omrxypos[0][i+1] - self.omrxypos[1][i] for i in range(peaknum-1)]
        mpw = int(np.mean(pw))
        mvw = int(np.mean(vw))
        # reduce wider peak
        for i in range(peaknum - 1):
            if pw[i] > mpw + 3:
                if vw[i] < mvw:
                    self.omrxypos[1][i] = self.omrxypos[1][i] - (pw[i] - mpw)
                    self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i]+mpw] = 0
                else:
                    self.omrxypos[0][i] = self.omrxypos[0][i] + (pw[i]- mpw)
                    self.xmap[self.omrxypos[0][i] - mpw:self.omrxypos[1][i]] = 0
        # move peak
        vw = [self.omrxypos[0][i+1] - self.omrxypos[1][i] for i in range(peaknum-1)]
        # mvw = int(np.mean(vw))
        for i in range(1, peaknum-1):
            # move left
            if vw[i-1] > vw[i] + 3:
                self.omrxypos[0][i] = self.omrxypos[0][i] - 3
                self.omrxypos[1][i] = self.omrxypos[1][i] - 3
                self.xmap[self.omrxypos[0][i]:self.omrxypos[0][i]+3] = 1
                self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i]+3] = 0
                if self.display:
                    print(f'move peak{i} to left')
            # move right
            if vw[i] > vw[i-1] + 3:
                self.omrxypos[0][i] = self.omrxypos[0][i] + 3
                self.omrxypos[1][i] = self.omrxypos[1][i] + 3
                self.xmap[self.omrxypos[0][i]-3:self.omrxypos[0][i]] = 0
                self.xmap[self.omrxypos[1][i]-3:self.omrxypos[1][i]] = 1
                if self.display:
                    print(f'move peak{i} to right')

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
        # fill sharp valley -0-
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
                print(f'start pos num({len(poslist[0])}) != end pos num({len(poslist[1])}):',
                      rowmark, step, count)
            return False
        # pos error: start pos less than end pos
        for pi in range(poslen):
            if poslist[0][pi] > poslist[1][pi]:
                if self.display:
                    print('start pos is less than end pos, in: ',
                          'horizon marks check' if rowmark else 'vertical marks check',
                          step, count)
                return False
        # width > 4 is considered valid mark block.
        tl = np.array([abs(x1 - x2) for x1, x2 in zip(poslist[0], poslist[1])])
        validnum = len(tl[tl > 4])
        setnum = self.omr_mark_area['mark_horizon_number'] \
                 if rowmark else \
                 self.omr_mark_area['mark_vertical_number']
        if validnum != setnum:
            if self.display:
                ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'valid mark num({validnum}) != mark_set_num({setnum}) in:',
                      ms, step, count)
            return False
        if len(tl) != setnum:
            if self.display:
                ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'checked marks num{len(tl)} != marks_set_num{setnum} in \
                direction={ms}, step={step}, count={count}')
            return False
        # max width is too bigger than min width is a error result. 20%(3-5 pixels)
        maxwid = max(tl); minwid = min(tl); widratio = minwid/maxwid
        if widratio < 0.2:
            if self.display:
                ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'maxwid/minwid = {maxwid}/{minwid} \
                     too big in {ms}, step={step}, count={count}')
            return False
        # check max gap between 2 peaks
        tc = np.array([poslist[0][i+1] - poslist[0][i] for i in range(poslen-1)])
        maxval = max(tc); minval = min(tc); gapratio = round(maxval/minval, 2)
        # r = round(gapratio, 2)
        if gapratio > 3:
            if self.display:
                ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'mark gap is singular! max/min = {gapratio} in \
                     {ms}, step={step},count={count}')
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

    def get_block_satu2(self, bmat, row, col):
        # return self.get_block_saturability(bmat)
        xs = self.omrxypos[2][row]; xe = self.omrxypos[3][row]+1
        ys = self.omrxypos[0][col]; ye = self.omrxypos[1][col]+1
        # origin
        sa = self.get_block_saturability(bmat)
        if sa[0] > 120:
            return sa
        # move left
        bmat =self.img[xs:xe, ys-2:ye-2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move right
        bmat =self.img[xs:xe, ys+2:ye+2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move up
        bmat =self.img[xs-2:xe-2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move down
        bmat =self.img[xs+2:xe+2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        return sa

    def get_block_saturability(self, blockmat):
        st0 = round(blockmat.mean(), 2)
        # visible pixel maybe 90 or bigger
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        th = self.omr_threshold
        r1 = len(rowmean[rowmean > th]) / len(rowmean)
        r2 = len(colmean[colmean > th]) / len(colmean)
        st1 = round(max(r1, r2),2)
        # the number of big pixel
        bignum = len(blockmat[blockmat > self.omr_threshold])
        st2 = round(bignum / blockmat.size, 2)
        # evenness
        m = 0, round(blockmat.shape[0] / 3), round(blockmat.shape[0] * 2/3), blockmat.shape[0]
        n = 0, round(blockmat.shape[1] / 3), round(blockmat.shape[1] * 2/3), blockmat.shape[1]
        st3 = 0
        for i in range(3):
            for j in range(3):
                st3 = st3 + (1 if blockmat[m[i]:m[i+1], n[j]:n[j+1]].mean() > self.omr_threshold \
                      else 0)
        st3 = round(st3 / 9, 2)
        # satu lines with big pixels much
        lth = 40
        st4 = sum([1 if len(np.where(blockmat[x, :] > lth)[0]) > 0.75 * blockmat.shape[1] else 0 \
                   for x in range(blockmat.shape[0])])
        st4 = 1 if st4 >= 3 else 0
        # st5 = sum([1 if len(np.where(blockmat[:, x] > lth)[0]) > 0.8 * blockmat.shape[0] else 0 \
        #           for x in range(blockmat.shape[1])])
        # st5 = 1 if st5 >= 4 else 0
        return st0/255, st1, st2, st3, st4

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
        marknum = len(self.omr_recog_data['label2'])
        for p in range(marknum):
            if self.omr_recog_data['label2'][p] == 1:
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

    # create recog_data, and test use svm in sklearn
    def get_recog_data(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        if lencheck == 0:
            if self.display:
                print('no position vector! cannot create recog_data[coord, \
                     data, label, saturation]!')
            return
        self.omr_recog_data = {'coord':[], 'label': [], 'bmean': [],  'satu':[]}
        total_mean = 0; pnum = 0
        for j in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                painted_mean = self.omrdict[(i, j)].mean()
                self.omr_recog_data['bmean'].append(round(painted_mean,2))
                self.omr_recog_data['coord'].append((i, j))
                self.omr_recog_data['satu'].append(
                    self.get_block_satu2(self.omrdict[(i, j)], i, j))
                total_mean = total_mean + painted_mean; pnum = pnum +1
        total_mean = total_mean / pnum
        self.omr_recog_data['label'] = [1 if x > total_mean else 0
                                        for x in self.omr_recog_data['bmean']]
        clu = KMeans(2)
        clu.fit(self.omr_recog_data['satu'])
        self.omr_recog_data['label2'] = clu.predict(self.omr_recog_data['satu'])

    def get_recog_markcoord(self):
        xylist = []
        if len(self.omr_recog_data['label']) == 0:
            print('recog data not created yet!')
            return
        # num = 0
        for coord, label in zip(self.omr_recog_data['coord'],
                                self.omr_recog_data['label']):
            if label == 1:
                xylist = xylist + [coord]
        return xylist

    def get_result_dataframe(self):
        f = self.fun_findfile(self.imgfile)
        rdf = pd.DataFrame({'card': [f] * len(self.omr_recog_data['label']),
                             'coord': self.omr_recog_data['coord'],
                             'label': self.omr_recog_data['label'],
                             'label2': self.omr_recog_data['label2'],
                             'bmean': self.omr_recog_data['bmean'],
                             'satu': self.omr_recog_data['satu']})
        # set label2 sign to 1 for painted (1 at max mean value)
        if rdf.sort_values('bmean', ascending=False).head(1)['label2'].values[0] == 0:
            rdf['label2'] = rdf['label2'].apply(lambda x: 1 - x)
        # rdf['label3'] = rdf['bmean'].apply(lambda x:1 if x > self.omr_threshold else 0)
        # rdf['label3'] = rdf['label'] + rdf['label2']
        # rdf['label3'] = rdf['label3'].apply(lambda x:0 if x == 2 else x)
        return rdf

    # --- some useful functions in omrmodel or outside
    @staticmethod
    def fun_show_image(fstr):
        if os.path.isfile(fstr):
            plt.imshow(mg.imread(fstr))
            plt.title(fstr)
            plt.show()
        else:
            print(f'no file={fstr}!')

    @staticmethod
    def fun_findfile(pathfile):
        ts = pathfile
        ts.replace('/', '\\')
        p1 = ts.find('\\')
        if p1 > 0:
            ts = ts[::-1]; p1=ts.find('\\'); ts = ts[0 : p1]; ts = ts[::-1]
        return ts

    @staticmethod
    def fun_findpath(pathfile):
        ts = OmrModel.fun_findfile(pathfile)
        return pathfile.replace(ts, '')

    # --- show omrimage or result data ---
    def plot_rawimage(self):
        plt.figure(1)
        plt.title(self.imgfile)
        plt.imshow(self.img)

    def plot_xmap(self):
        plt.figure(2)
        plt.plot(self.xmap)

    def plot_ymap(self):
        plt.figure(3)
        plt.plot(self.ymap)

    def plot_mark_omrimage(self):
        if type(self.mark_omriamge) != np.ndarray:
            #print('mark omr image is not created!')
            self.get_mark_omrimage()
        plt.figure(4)
        plt.title('recognized - omr - region ' + self.imgfile)
        plt.imshow(self.mark_omriamge)

    def plot_recog_omrimage(self):
        if type(self.recog_omriamge) != np.ndarray:
            #print('mark omr image is not created!')
            self.get_recog_omrimage()
        plt.figure(5)
        plt.title('recognized - omr - region' + self.imgfile)
        plt.imshow(self.recog_omriamge)

    def plot_omrblock_mean(self):
        plt.figure(6)
        plt.title(self.imgfile)
        data = self.omr_recog_data['mean'].copy()
        data.sort()
        plt.plot([x for x in range(len(data))], data)
