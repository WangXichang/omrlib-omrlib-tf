# *_* utf-8 *-*


import numpy as np
import pandas as pd
import matplotlib.image as mg
import matplotlib.pyplot as plt
# from sklearn import svm
from sklearn.cluster import KMeans
# import cv2
import time
import os
from scipy.ndimage import filters
import tensorflow as tf


def readomr_task(cardno):
    fpath, cardformat, group, flist = card(cardno)
    omr = OmrModel()
    omr.set_format(cardformat)
    omr.set_group(group)
    omr.debug = False  # output omr string only
    # print(omr.omr_group_map)
    readomr_result = None
    sttime = time.clock()
    runcount = 0
    for f in flist:
        print(round(runcount/len(flist), 2), '-->', f)
        omr.set_img(f)
        omr.run()
        if runcount == 0:
            readomr_result = omr.get_result_dataframe()
        else:
            readomr_result = readomr_result.append(omr.get_result_dataframe())
        # omr.save_result_omriamge()
        runcount += 1
    print(time.clock()-sttime, '\n', runcount)
    rg = readomr_result.sort_values(['card', 'group', 'coord']).\
        groupby(['card'])[['code']].sum()
    rg['pnum'] = rg['code'].apply(lambda x: len(x.replace('.', '')))
    return readomr_result, rg


def test_one(fname:str, cardformat: tuple, cardgroup: dict, display=True):
    omr = OmrModel()
    omr.imgfile = fname
    omr.set_format(cardformat)
    omr.set_group(cardgroup)
    omr.display = display
    omr.debug = True
    omr.run()
    r = omr.get_result_dataframe()
    return omr, r


def card(no):
    filter_file: list = ['.jpg']
    f_path: str = ''
    data_source: str = 'surface'  # '3-2'  #
    card_format: list = []
    group_dict = {}
    if no == 1:
        f_path = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\' \
                if data_source != '3-2' else \
                'F:\\studies\\juyunxia\\omrimage1\\'
        card_format = [37, 14, 23, 36, 1, 13]
        group_dict = {j: [(1, 23+j-1), 10, 'V', '0123456789', 'S'] for j in range(1,15)}
    elif no == 2:  # OMR..jpg
        filter_file = filter_file + ['OMR']
        f_path = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\' \
                if data_source != '3-2' else \
                'F:\\studies\\juyunxia\\omrimage2\\'
        card_format = [31, 6, 1, 30, 1, 5]
        group_dict = {i + j*5: [(i, 2 + j*6), 4, 'H', 'ABCD', 'S'] for i in range(1,6) \
                      for j in range(5)}
    elif no == 3:  # Oomr..jpg
        filter_file = filter_file + ['Oomr']
        f_path = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\' \
                if data_source != '3-2' else \
                'F:\\studies\\juyunxia\\omrimage2\\'
        card_format = [20, 11, 1, 19, 1, 10]
        group_dict = {i: [(1, i), 10, 'V', '0123456789', 'S'] for i in range(1,20)}
    elif no == 101:
        filter_file = filter_file + ['1-']
        f_path = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr0\\' \
            if data_source != '3-2' else \
            'F:\\studies\\juyunxia\\omrimage2\\'
        card_format = [25, 8, 2, 24, 3, 7]
        group_dict = {i-2: [(i, 3), 4, 'H', 'ABCD', 'S'] for i in range(3,8)}
        group_dict.update({i+5-2: [(i, 9), 4, 'H', 'ABCD', 'S'] for i in range(3,8)})
        group_dict.update({i+10-2:[(i, 15), 4, 'H', 'ABCD', 'S'] for i in range(3,8)})
        group_dict.update({16:[(3, 21), 4, 'H', 'ABCD', 'S']})
    elif no == 102:
        filter_file = filter_file + ['2-']
        f_path = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr0\\' \
            if data_source != '3-2' else \
            'F:\\studies\\juyunxia\\omrimage2\\'
        card_format = [25, 8, 2, 24, 3, 7]
        group_dict = {i-2: [(i, 3), 4, 'H', 'ABCD', 'S'] for i in range(3,8)}
        group_dict.update({i+5-2: [(i, 9), 4, 'H', 'ABCD', 'S'] for i in range(3,8)})
        group_dict.update({i+10-2:[(i, 15), 4, 'H', 'ABCD', 'S'] for i in range(3,8)})
        group_dict.update({i+15-2:[(i, 21), 4, 'H', 'ABCD', 'S'] for i in range(3,8)})
    file_list = []
    for dir_path, dir_names, file_names in os.walk(f_path):
        for file in file_names:
            b = True
            for ss in filter_file:
                b = b & (ss in file)
            if b:
                file_list.append(os.path.join(dir_path, file))
    return [f_path, card_format, group_dict, file_list]


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
        # omr parameters
        self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        self.omr_group: dict = {1: [(0, 0), 4, 'H', 'ABCD', 'S']}  # pos, len, dir, code, mode
        self.omr_group_map: dict = {}
        # system control parameters
        self.debug: bool = False
        self.display: bool = False        # display time, error messages in running process
        self.logwrite: bool = False       # record processing messages in log file, finished later
        # inner parameter
        self.omr_threshold: int = 60
        self.check_vertical_window: int = 30
        self.check_horizon_window: int = 20
        self.check_step: int = 10
        # result data
        self.xmap: list = []
        self.ymap: list = []
        self.omrxypos: list = [[], [], [], []]
        self.mark_omriamge = None
        self.recog_omriamge = None
        self.omrdict: dict = {}
        self.omr_recog_data: dict = {}
        # self.omrsvm = None

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
        if self.display:
            print(f'consume {time.clock()-st}')

    def set_format(self, card_form: tuple):
        """
        :param
            card_form = [mark_h_num, mark_v_num,
                         valid_h_start, valid_h_end,
                         valid_v_start, valid_v_end]
        :return
            False and error messages if set data is error
            for example: valid position is not in mark area
        """
        if (card_form[2] < 1) | (card_form[3] < card_form[2]) | (card_form[3] > card_form[0]):
            print(f'mark setting error: mark start{card_form[2]}, end{card_form[3]}')
            return
        if (card_form[4] < 1) | (card_form[5] < card_form[4]) | (card_form[5] > card_form[1]):
            print(f'mark setting error: mark start{card_form[2]}, end{card_form[3]}')
            return
        self.omr_mark_area['mark_horizon_number'] = card_form[0]
        self.omr_mark_area['mark_vertical_number'] = card_form[1]
        self.omr_valid_area['mark_horizon_number'] = [card_form[2], card_form[3]]
        self.omr_valid_area['mark_vertical_number'] = [card_form[4], card_form[5]]

    def set_group(self, group: dict):
        """
        :param group: {g_no:[g_pos(row, col), g_len:int, g_direction:'V' or 'H', g_codestr:str,
                       g_mode:'S'/'M'], ...}
        g_no:int,  serial No for groups
        g_pos: (row, col), 1 ... maxno, start coordinate for each painting group,
        g_len: length for each group
        g_codestr: code string for painting block i.e. 'ABCD', '0123456789'
        g_direction: 'V' or 'H' for vertical or hironal direction
        g_mode: 'S' or 'M' for single or multi-choice
        """
        self.omr_group = group
        for gno in group.keys():
            for j in range(self.omr_group[gno][1]):
                # get pos coordination (row, col)
                x, y = self.omr_group[gno][0]
                # add -1 to set to 0 ... n-1 mode
                x, y = (x+j-1, y-1) if self.omr_group[gno][2] == 'V' else (x-1, y+j-1)
                # create (x, y):[gno, code, mode]
                self.omr_group_map[(x, y)] = (gno, self.omr_group[gno][3][j], self.omr_group[gno][4])
                # check (x, y) in mark area
                hscope = self.omr_valid_area['mark_horizon_number']
                vscope = self.omr_valid_area['mark_vertical_number']
                if (y not in range(hscope[1])) | (x not in range(vscope[1])):
                    print(f'group set error: ({x}, {y}) not in valid mark area{vscope}, {hscope}!')

    def set_img(self, fname: str):
        self.imgfile = fname

    def get_img(self, imfile):
        self.img = 255 - mg.imread(imfile)  # type: np.array
        # self.img = np.array(self.img)
        if len(self.img.shape) == 3:
            self.img = self.img.mean(axis=2)

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
                    print(f'{direction}: marks not found in direction',
                          f'imagezone= {maxlen- w - step*count}:{maxlen  - step*count}',
                          f'count={count}, step={step}, window={window}!')
                break
            imgmap = img[maxlen - w - step * count:maxlen - step * count, :].sum(axis=0) \
                if rowmark else \
                img[:, maxlen - w - step * count:maxlen - step * count].sum(axis=1)
            mark_start_end_position = self.check_mark_block(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    if self.display:
                        direction = 'horizon' if rowmark else 'vertical'
                        print(f'{direction}: mark result，step={step},count={count}',
                              f'imgzone={maxlen - w - step * count}:{maxlen - step * count}',
                              f'mark={len(mark_start_end_position[0])}')
                    return mark_start_end_position, step, count
            count += 1
        # print(f'no correct mark position solution found, row={rowmark}, step={step}, count={count}')
        return mark_start_end_position, step, count

    def check_mark_block(self, mapvec, rowmark) -> tuple:
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
        result = np.where(r1 == judg_value)[0] + 2, np.where(r2 == judg_value)[0] + 2
        return result

    def check_mark_adjustpeak(self):
        # return
        peaknum = len(self.omrxypos[0])
        pw = np.array([self.omrxypos[1][i] - self.omrxypos[0][i] for i in range(peaknum)])
        vw = np.array([self.omrxypos[0][i+1] - self.omrxypos[1][i] for i in range(peaknum-1)])
        mpw = int(pw.mean())
        mvw = int(vw.mean())
        # reduce wider peak
        for i in range(peaknum - 1):
            if pw[i] > mpw + 3:
                if vw[i] < mvw:
                    self.omrxypos[1][i] = self.omrxypos[1][i] - (pw[i] - mpw)
                    self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i] + mpw] = 0
                else:
                    self.omrxypos[0][i] = self.omrxypos[0][i] + (pw[i] - mpw)
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
                self.xmap[self.omrxypos[1][i]:self.omrxypos[1][i]+3+1] = 0
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

    @staticmethod
    def check_mark_mapfun_smoothsharp(mapf):
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
        window = self.check_horizon_window if rowmark else self.check_vertical_window
        imgwid = self.img.shape[0] if rowmark else self.img.shape[1]
        hvs = 'horizon:' if rowmark else 'vertical:'
        # start position number is not same with end posistion number
        if poslen != len(poslist[1]):
            if self.display:
                print(f'{hvs} start pos num({len(poslist[0])}) != end pos num({len(poslist[1])})',
                      f'step={step},count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # pos error: start pos less than end pos
        for pi in range(poslen):
            if poslist[0][pi] > poslist[1][pi]:
                if self.display:
                    print(f'{hvs} start pos is less than end pos, step={step},count={count}',
                          f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
                return False
        # width > 4 is considered valid mark block.
        tl = np.array([abs(x1 - x2) for x1, x2 in zip(poslist[0], poslist[1])])
        validnum = len(tl[tl > 4])
        setnum = self.omr_mark_area['mark_horizon_number'] \
            if rowmark else \
            self.omr_mark_area['mark_vertical_number']
        if validnum != setnum:
            if self.display:
                # ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} mark valid num({validnum}) != set_num({setnum})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        if len(tl) != setnum:
            if self.display:
                print(f'{hvs}checked mark num({len(tl)}) != set_num({setnum})',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # max width is too bigger than min width is a error result. 20%(3-5 pixels)
        maxwid = max(tl)
        minwid = min(tl)
        widratio = minwid/maxwid
        if widratio < 0.2:
            if self.display:
                ms = 'horizon marks check' if rowmark else 'vertical marks check'
                print(f'{hvs} maxwid/minwid = {maxwid}/{minwid}',
                      f'step={step}, count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        # check max gap between 2 peaks
        p1, p2 = self.omr_valid_area['mark_horizon_number'] \
                 if rowmark else \
                 self.omr_valid_area['mark_vertical_number']
        tc = np.array([poslist[0][i+1] - poslist[0][i] for i in range(p1-1, p2)])
        # tc = np.array([poslist[0][i+1] - poslist[0][i] for i in range(poslen-1)])
        maxval = max(tc)
        minval = min(tc)
        gapratio = round(maxval/minval, 2)
        # r = round(gapratio, 2)
        if gapratio > 3:
            if self.display:
                print(f'{hvs} mark gap is singular! max/min = {gapratio}',
                      f'step={step},count={count}',
                      f'imagezone={imgwid - window - count*step}:{imgwid - count*step}')
            return False
        return True

    def get_omrdict_xyimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('no position vector created! omrdict cteated fail!')
            return
        # cut area for painting points
        for x in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                self.omrdict[(y, x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                                self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_block_satu2(self, bmat, row, col):
        # return self.get_block_saturability(bmat)
        xs = self.omrxypos[2][row]
        xe = self.omrxypos[3][row]+1
        ys = self.omrxypos[0][col]
        ye = self.omrxypos[1][col]+1
        # origin
        sa = self.get_block_saturability(bmat)
        if sa[0] > 120:
            return sa
        # move left
        bmat = self.img[xs:xe, ys-2:ye-2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move right
        bmat = self.img[xs:xe, ys+2:ye+2]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move up
        bmat = self.img[xs-2:xe-2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move down
        bmat = self.img[xs+2:xe+2, ys:ye]
        sa2 = self.get_block_saturability(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        return sa

    def get_block_saturability(self, blockmat):
        # get 0-1 image with threshold
        block01 = self.fun_normto01(blockmat, self.omr_threshold)
        # feature1: mean level
        # use coefficient 10/255 normalizing
        st0 = round(blockmat.mean()/255*10, 2)
        # feature2: big-mean-line_ratio in row or col
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        th = self.omr_threshold
        r1 = len(rowmean[rowmean > th]) / len(rowmean)
        r2 = len(colmean[colmean > th]) / len(colmean)
        st1 = round(max(r1, r2), 2)
        # feature3: big-pixel-ratio
        bignum = len(blockmat[blockmat > self.omr_threshold])
        st2 = round(bignum / blockmat.size, 2)
        # feature4: hole-number
        st3 = self.fun_detect_hole(block01)
        # saturational area is more than 3
        th = self.omr_threshold  #50
        # feature5: saturation area exists
        # st4 = cv2.filter2D(p, -1, np.ones([3, 5]))
        st4 = filters.convolve(self.fun_normto01(blockmat, th),
                               np.ones([3, 5]), mode='constant')
        st4 = 1 if len(st4[st4 >= 14]) >= 1 else 0
        return st0, st1*2, st2, st3, st4

    @staticmethod
    def fun_detect_hole(mat):
        r1 = 0
        r2 = 0
        # 3x5 hole
        m = np.ones([3, 5]); m[1, 1:4] = -1
        rf = filters.convolve(mat, m, mode='constant')
        r1 = len(rf[rf == 12])
        if r1 == 0:
            m = np.ones([3, 5]); m[1, 1:3] = -1; m[1,3] = 0
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        if r1 ==0:
            m = np.ones([3, 5]); m[1, 2:4] = -1; m[1, 1] = 0
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        # 4x5 hole
        m = np.ones([4, 5]); m[1, 1:4] = -1; m[2, 1:4] = -1
        m[1:3,1] = 0; m[0, 0] = 0; m[0, 4] = 0; m[3, 0] = 0; m[3, 4] = 0
        rf = filters.convolve(mat, m, mode='constant')
        r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.ones([4, 5]); m[1, 1:4] = -1; m[2, 1:4] = -1
            m[1:3, 3] = 0; m[0, 0] = 0; m[0, 4] = 0; m[3, 0] = 0; m[3, 4] = 0
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.ones([4, 5]); m[1, 1:4] = -1; m[2, 1:4] = -1
            m[1, 1:4] = 0; m[0, 0] = 0; m[0, 4] = 0; m[3, 0] = 0; m[3, 4] = 0
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.ones([4, 5]); m[1, 1:4] = -1; m[2, 1:4] = -1
            m[2, 1:4] = 0; m[0, 0] = 0; m[0, 4] = 0; m[3, 0] = 0; m[3, 4] = 0
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        return (1 if r1 > 0 else 0) + (1 if r2 >0 else 0)

    @staticmethod
    def fun_normto01(mat, th):
        m = np.copy(mat)
        m[m < th] = 0
        m[m >= th] = 1
        return m

    @staticmethod
    def fun_save_omr_tfrecord(tfr_pathfile: str, dataset_labels: list, dataset_images: dict):
        """
        function:
            save omr_dict to tfrecord file{features:label, painting_block_image}
        parameters:
            param tfr_pathfile: lcoation+filenamet to save omr label+blockimage
            param dataset_labels: key(coord)+label(str), omr block image label ('0'-unpainted, '1'-painted)
            param dataset_images: key(coord):blockiamge
        return:
            TFRecord file= [tfr_pathfile].tfrecord
        """
        sess = tf.Session()
        writer = tf.python_io.TFRecordWriter(tfr_pathfile+'.tfrecord')
        for key, label in dataset_labels:
            omr_image = dataset_images[key]
            omr_image3 = omr_image.reshape([omr_image.shape[0], omr_image.shape[1], 1])
            resized_image = tf.image.resize_images(omr_image3, [12, 15])
            #resized_image = omr_image
            bytes_image = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            omr_label = label.encode('utf-8')
            example = tf.train.Example(features=tf.train.Features(feature= {
                'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[omr_label])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))
                }))
            writer.write(example.SerializeToString())
        writer.close()
        sess.close()

    @staticmethod
    def fun_read_omr_tfrecord(tfr_pathfile):
        omr_data_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(tfr_pathfile)
            )
        sess = tf.Session()
        reader = tf.TFRecordReader()
        _, ser = reader.read(omr_data_queue)
        omr_data = tf.parse_single_example(ser,
                    features={
                        'label': tf.FixedLenFeature([], tf.string),
                        'image': tf.FixedLenFeature([], tf.string),
                    })
        omr_image = tf.decode_raw(omr_data['image'], tf.uint8)
        omr_image_reshape = tf.reshape(omr_image, [12, 15, 1])
        omr_label = tf.cast(omr_data['label'], tf.string)
        return omr_image_reshape, omr_label

    def get_mark_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | (len(self.omrxypos[0]) != len(self.omrxypos[3])) | \
                       (len(self.omrxypos[1]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('no position vector created! so cannot create omrdict!')
            return
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for col in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                         self.omr_valid_area['mark_horizon_number'][1]):
            for row in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                             self.omr_valid_area['mark_vertical_number'][1]):
                omrimage[self.omrxypos[2][row]: self.omrxypos[3][row]+1,
                         self.omrxypos[0][col]: self.omrxypos[1][col]+1] = self.omrdict[(row, col)]
        self.mark_omriamge = omrimage

    def get_recog_omrimage(self):
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        valid_result = (lencheck == 0) | (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                       (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if valid_result:
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

    # create recog_data, and test use svm in sklearn
    def get_recog_data(self):
        # self.omr_recog_data = {'coord': [], 'label': [], 'bmean': [],  'satu': []}
        self.omr_recog_data = {'coord': [], 'feature': [], 'group':[], 'code':[], 'mode':[], 'label':[]}
        lencheck = len(self.omrxypos[0]) * len(self.omrxypos[1]) * \
                   len(self.omrxypos[3]) * len(self.omrxypos[2])
        invalid_result = (lencheck == 0) | (len(self.omrxypos[0]) != len(self.omrxypos[1])) | \
                       (len(self.omrxypos[2]) != len(self.omrxypos[3]))
        if invalid_result:
            if self.display:
                print('no position vector! cannot create recog_data[coord, \
                     data, label, saturation]!')
            return
        # total_mean = 0
        # pnum = 0
        for j in range(self.omr_valid_area['mark_horizon_number'][0]-1,
                       self.omr_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_valid_area['mark_vertical_number'][0]-1,
                           self.omr_valid_area['mark_vertical_number'][1]):
                # painted_mean = self.omrdict[(i, j)].mean()
                # self.omr_recog_data['bmean'].append(round(painted_mean, 2))
                self.omr_recog_data['coord'].append((i, j))
                # whether using moving block by satu2
                self.omr_recog_data['feature'].append(
                    self.get_block_saturability(self.omrdict[(i, j)]))
                # self.omr_recog_data['feature'].append(
                #    self.get_block_satu2(self.omrdict[(i, j)], i, j))
                self.omr_recog_data['group'].append(( \
                    self.omr_group_map[(i,j)][0] if (i,j) in self.omr_group_map else -1))
                self.omr_recog_data['code'].append(( \
                    self.omr_group_map[(i,j)][1] if (i,j) in self.omr_group_map else '.'))
                self.omr_recog_data['mode'].append(( \
                    self.omr_group_map[(i,j)][2] if (i,j) in self.omr_group_map else '.'))
                # total_mean = total_mean + painted_mean
                # pnum = pnum + 1
        # total_mean = total_mean / pnum
        # self.omr_recog_data['label'] = [1 if x > total_mean else 0
        #                                for x in self.omr_recog_data['bmean']]
        clu = KMeans(2)
        clu.fit(self.omr_recog_data['feature'])
        self.omr_recog_data['label'] = clu.predict(self.omr_recog_data['feature'])

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
        if len(self.omr_recog_data['label']) == 0:
            print('no recog data created!')
            return pd.DataFrame(self.omr_recog_data)
        f = self.fun_findfile(self.imgfile)
        rdf = pd.DataFrame({'card': [f] * len(self.omr_recog_data['label']),
                            'coord': self.omr_recog_data['coord'],
                            'label': self.omr_recog_data['label'],
                            'feat': self.omr_recog_data['feature'],
                            'group': self.omr_recog_data['group'],
                            'code': self.omr_recog_data['code'],
                            'mode': self.omr_recog_data['mode'],
                            })
        # 'label': self.omr_recog_data['label'],
        # 'bmean': self.omr_recog_data['bmean'],
        # set label2 sign to 1 for painted (1 at max mean value)
        # rdf['label3'] = rdf['bmean'].apply(lambda x:1 if x > self.omr_threshold else 0)
        # rdf['label3'] = rdf['label'] + rdf['label2']
        # rdf['label3'] = rdf['label3'].apply(lambda x:0 if x == 2 else x)
        if rdf.sort_values('feat', ascending=False).head(1)['label'].values[0] == 0:
            rdf['label'] = rdf['label'].apply(lambda x: 1 - x)
        # rdf.code = [rdf.code[i] if rdf.label[i] == 1 else '.' for i in range(len(rdf.code))]
        rdf.loc[rdf.label == 0, 'code'] = '.'
        if not self.debug:
            r = rdf[rdf.group > 0][['card', 'group', 'coord', 'code', 'mode']]
            # r['codelen'] = r.code.apply(lambda s: len(s.replace('.', '')))
            return r.sort_values('group')
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
            ts = ts[::-1]
            p1 = ts.find('\\')
            ts = ts[0: p1]
            ts = ts[::-1]
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
            self.get_mark_omrimage()
        plt.figure(4)
        plt.title('recognized - omr - region ' + self.imgfile)
        plt.imshow(self.mark_omriamge)

    def plot_recog_omrimage(self):
        if type(self.recog_omriamge) != np.ndarray:
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
