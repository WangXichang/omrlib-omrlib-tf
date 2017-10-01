# *_* utf-8 *-*


import numpy as np
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
    # fpath = r'C:\Users\wangxichang\students\ju\testdata\omr1'     # surface data
    # fpath = r'C:\Users\wangxichang\students\ju\testdata\omr2'     # surface data
    fpath = r'f:\studies\juyunxia\omrimage1'      # 3-2 data
    # fpath = r'f:\studies\juyunxia\omrimage2'      # 3-2 data
    omr.omr_valid_area = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [22, 36]}
    omr.omr_set_vertical_number = 14
    omr.omr_set_horizon_number = 37
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
        readomr_result[f] = omr.omr_recog_data
        runcount += 1
    print(time.clock()-sttime, '\n', runcount)
    for k in readomr_result:
        if len(readomr_result[k]) != 14:
            print(k, len(readomr_result[k]))
    return readomr_result

def test(filename=''):
    omr = OmrRecog()
    if len(filename) == 0:
        # omrfile = 'omrtest3.jpg'
        # omrfile = r'C:\Users\wangxichang\students\ju\testdata\omr1\B84261310881012002_Omr01.jpg'
        # card
        # omrfile = r'f:\studies\juyunxia\omrimage1\B84261310881012002_Omr01.jpg'  # 3-2 data
        # self.omr_mark_area = {'mark_horizon_number': 37, 'mark_vertical_number': 14}
        # self.omr_valid_area = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [23, 36]}
        # card
        omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_Oomr01.jpg'  # 3-2 data
        omr.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        omr.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
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
        self.omr_threshold = 60
        self.omr_mean_threshold = 100
        self.check_vertical_window = 30
        self.check_horizon_window = 30
        self.check_step = 5
        # result data
        self.omr_result = []
        self.xmap = None
        self.ymap = None
        self.omrxypos = [[],[],[],[]]
        self.mark_omriamge = None
        self.recog_omriamge = None
        self.omrdict = {}
        self.omr_recog_data = {}
        self.omrsvm = None

    def test(self,filename=''):
        if len(filename) == 0:
            omrfile = 'omrtest3.jpg'
            # card
            # omrfile = r'C:\Users\wangxichang\students\ju\testdata\omr1\B84261310881012002_Omr01.jpg'
            # omrfile = r'f:\studies\juyunxia\omrimage1\B84261310881012002_Omr01.jpg'  # 3-2 data
            # self.omr_mark_area = {'mark_horizon_number': 37, 'mark_vertical_number': 14}
            # self.omr_valid_area = {'mark_horizon_number': [1, 13], 'mark_vertical_number': [23, 36]}
            # card
            # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_Oomr01.jpg'  # 3-2 data
            self.omr_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
            self.omr_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
            # card
            # omrfile = r'f:\studies\juyunxia\omrimage2\1a3119261913111631103_OMR01.jpg'  # 3-2 data
        else:
            omrfile = filename
        self.imgfile = omrfile
        self.run()

    def run(self):
        # initiate some variables
        self.omrxypos = [[],[],[],[]]
        # start running
        st = time.clock()
        self.get_img(self.imgfile)
        self.get_markblock()
        self.get_omrdict_xyimage()
        self.get_mark_omrimage()
        self.get_recog_data()
        self.get_recog_omrimage()
        print(f'consume {time.clock()-st}')

    def set_imgfilename(self, fname):
        self.imgfile = fname

    def get_img(self, imfile):
        self.img = 255 - mg.imread(imfile)
        if len(self.img.shape) == 3:
            self.img = self.img.mean(axis=2)

    def get_markblock(self):
        # r1 = [[],[]]; r2 = [[], []]
        # check horizonal mark blocks (columns number)
        r1, _step, _count = self.check_mark_pos(self.img, rowmark=True, step=3, window=30)
        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom as img row bottom to remove noise
        rownum = self.img.shape[0]
        rownum = rownum - _step * _count
        r2, step, count = self.check_mark_pos(self.img[0:rownum, :], \
                                              rowmark=False, step=5, window=30)
        if (len(r1[0]) > 0) | (len(r2[0]) > 0):
            self.omrxypos = np.array([r1[0], r1[1], r2[0], r2[1]])

    def check_mark_pos(self, img, rowmark, step, window):
        w = window
        vpol = self.img.shape[0] if rowmark else self.img.shape[1]
        mark_start_end_position = [[],[]]
        count = 0
        while True:
            # no mark area found
            if vpol < w + step * count:
                print('check mark block in direction=', 'horizon' if rowmark else 'vertical')
                print('no goog pos found, vpol, count,step,window = ',vpol, count, step, window)
                break
            imgmap =  img[vpol - w - step * count:vpol - step * count, :].sum(axis=0) \
                          if rowmark else \
                          img[:, vpol - w - step * count:vpol - step * count].sum(axis=1)
            mark_start_end_position = self.check_mark_block(imgmap, rowmark)
            if self.check_mark_result_evaluate(rowmark, mark_start_end_position, step, count):
                    # & self.evaluate_position_list(mark_start_end_position):
                    if self.display:
                        print('result:row={0},step={1},count={2}, imagescope={3}:{4}, marknum={5}'. \
                          format(rowmark, step, count, vpol - w - step * count, vpol - step * count,
                                 len(mark_start_end_position[0])))
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
        # start position number is not same with end posistion number
        if len(poslist[0]) != len(poslist[1]):
            if self.display:
                print(f'startposnum{len(poslist[0])} != endposnum{len(poslist[1])}:', rowmark, step, count)
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
        if widratio > 0.8:
            if self.display:
                print(f'maxwid{maxwid} vs minwid{minwid} too big in:', widratio, rowmark, step, count)
            return False
        return True

    def get_omrdict_xyimage(self):
        # cut area for painting points
        for x in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                self.omrdict[(y, x)] = self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                                                self.omrxypos[0][x]:self.omrxypos[1][x]+1]

    def get_mark_omrimage(self):
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.img.shape)
        for col in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for row in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                omrimage[self.omrxypos[2][row]: self.omrxypos[3][row]+1,
                         self.omrxypos[0][col]: self.omrxypos[1][col]+1] = self.omrdict[(row, col)]
        self.mark_omriamge = omrimage

    def get_recog_omrimage(self):
        recogomr = np.zeros(self.img.shape)
        marknum = len(self.omr_recog_data['label'])
        for p in range(marknum):
            if self.omr_recog_data['label'][p] == 1:
                _x, _y = self.omr_recog_data['coord'][p]
                h1, h2 = [self.omr_valid_area['mark_horizon_number'][0] - 1, self.omr_valid_area['mark_horizon_number'][1]]
                v1, v2 = [self.omr_valid_area['mark_vertical_number'][0] - 1, self.omr_valid_area['mark_vertical_number'][1]]
                if (_x in range(v1, v2)) & (_y in range(h1, h2)):
                    recogomr[self.omrxypos[2][_x]: self.omrxypos[3][_x]+1,
                             self.omrxypos[0][_y]: self.omrxypos[1][_y]+1] \
                        = self.omrdict[(_x, _y)]
            p += 1
        self.recog_omriamge = recogomr
        # print(omr.omr_svmdata['label'][p],omr.omr_svmdata['coord'][p])

    # test use svm in sklearn
    def get_recog_data(self):
        self.omr_recog_data = {'data': [], 'label': [], 'coord':[]}
        for j in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                painted_mean = self.omrdict[(i, j)].mean()
                painted_std0 = self.omrdict[(i, j)].mean(axis=0).std()
                painted_std1 = self.omrdict[(i, j)].mean(axis=1).std()
                self.omr_recog_data['data'].append([painted_mean, painted_std0, painted_std1])
                # self.omr_svmdata['data'].append([painted_mean, painted_std0])
                if painted_mean >= self.omr_mean_threshold:
                    # print(f'painted points: {(i, j)}')
                    self.omr_recog_data['label'].append(1)
                else:
                    self.omr_recog_data['label'].append(0)
                self.omr_recog_data['coord'].append((i, j))
        # clf = svm.LinearSVC()
        # clf.fit(self.omr_recog_data['data'], self.omr_recog_data['label'])
        # self.omrsvm = clf

    # deprecated function
    def get_omr_result(self):
        # cut area for painting points and set result to omr_result
        self.omr_result = []
        for y in range(self.omr_valid_area['mark_horizon_number'][0]-1, self.omr_valid_area['mark_horizon_number'][1]):
            for x in range(self.omr_valid_area['mark_vertical_number'][0]-1, self.omr_valid_area['mark_vertical_number'][1]):
                if self.img[self.omrxypos[2][y]:self.omrxypos[3][y]+1,
                   self.omrxypos[0][x]:self.omrxypos[1][x]+1].mean() \
                        > self.omr_threshold:
                    self.omr_result.append((y+1, x+1))

    def get_recog_markcoord(self):
        if len(self.omr_recog_data['label']) == 0:
            print('no recog data!')
            return
        num = 0
        for coord, data, label in zip(self.omr_recog_data['coord'], self.omr_recog_data['data'], self.omr_recog_data['label']):
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

    def plot_mean_colstd_scatter(self):
        plt.figure(6)
        plt.scatter([x[0] for x in self.omr_recog_data['data']],
                    [x[1] for x in self.omr_recog_data['data']])

    def plot_mean_rowstd_scatter(self):
        plt.figure(7)
        plt.scatter([x[0] for x in self.omr_recog_data['data']],
                    [x[2] for x in self.omr_recog_data['data']])
