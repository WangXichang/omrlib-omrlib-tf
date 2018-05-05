# *-* utf-8 *-*
# python 3.6x


import time
import os
import sys
import copy
import glob
import pprint as pp
import numpy as np
import pandas as pd
import matplotlib.image as mg
import matplotlib.pyplot as plt
from collections import Counter, namedtuple
from scipy.ndimage import filters
from sklearn.cluster import KMeans
from sklearn.externals import joblib as jb
import cv2

import warnings
warnings.simplefilter('error')
# import traceback
# import tensorflow as tf


def read_batch(former, data2file=''):
    """
    :input
        card_form: form(dict)/former(Former), could get from class OmrForm
        to_file: file name to save data, auto added .csv, if to_file=='' then not to save
    :return:
        omr_result_dataframe:
            card,         # file name
            len,          # result string length, no blank(no painted) blocks included
            result,       # recognized code string
            result_info,  # error painting info 'group_no:[painting result]'
            score,        # total score for card
            score_group,  # scores for group
    """

    if not isinstance(former, dict):
        if isinstance(former.form, dict):
            former = former.form
        else:
            print('invalid card form!')
            return

    if len(data2file) > 0:
        fpath = Util.find_path_from_pathfile(data2file)
        if not os.path.isdir(fpath):
            print('invaild path: ' + fpath)
            return
        no = 1
        while os.path.isfile(data2file + '.csv'):
            data2file += '_' + str(no)
            no += 1
        data2file += '.csv'

    # set model
    omr = OmrModel()
    omr.set_form(former)
    image_list = former['image_file_list']
    if len(image_list) == 0:
        print('no file found in card_form.image_file_list !')
        return None

    # run model
    omr_result = None
    sttime = time.clock()
    run_len = len(image_list)
    run_count = 0
    progress = ProgressBar(total=run_len)
    for f in image_list:
        omr.set_omr_image_filename(f)
        omr.run()
        rf = omr.omr_result_dataframe
        if run_count == 0:
            omr_result = rf
        else:
            omr_result = omr_result.append(rf)
        omr.card_index_no = run_count + 1
        run_count += 1
        progress.move()
        if run_count % 5 == 0:
            progress.log(f)
        progress.log(f)
    total_time = round(time.clock()-sttime, 2)
    if run_len != 0:
        print('total_time= %2.4f  mean_time= %2.2f' % (total_time, round(total_time / run_len, 2)))
        if len(data2file) > 0:
            omr_result.to_csv(data2file, columns=['card', 'valid', 'result', 'len', 'group'])
    return omr_result


def read_test(former,
              readfile='',
              display=True
              ):
    if hasattr(former, "form"):
        former = former.form
    elif not isinstance(former, dict):
        print('card_form is not dict!')
        return
    if len(readfile) == 0:
        if len(former['image_file_list']) > 0:
            readfile = former['image_file_list'][0]
        else:
            print('card_form do not include any image files!')
            return
    if not os.path.isfile(readfile):
        print('%s does not exist!' % readfile)
        return
    this_form = copy.deepcopy(former)
    this_form['image_file_list'] = [readfile]

    omr = OmrModel()
    omr.set_form(this_form)
    omr.set_omr_image_filename(readfile)

    omr.sys_run_test = True
    omr.sys_run_check = False
    omr.sys_display = display

    # try:
    omr.run()
    # except:
    #    print(traceback.format_exc())

    return omr


def read_check(
        readfile='',
        form2file='',
        clip_top=0,
        clip_bottom=0,
        clip_right=0,
        clip_left=0,
        detect_mark_step_length=3,
        detect_mark_max_stepnum=30,
        detect_mark_min_marknum=5,
        detect_mark_horizon_window=12,
        detect_mark_vertical_window=15,
        display_figures=True
        ):

    # init check mark location
    check_mark_fromright = True,
    check_mark_frombottom = True,

    # get image file
    if hasattr(readfile, "form"):
        if isinstance(readfile.form, dict):
            if 'image_file_list' in readfile.form.keys():
                if len(readfile.form['image_file_list']) > 0:
                    readfile = readfile.form['image_file_list'][0]
                else:
                    print('card_file[image_file_list] include no files!')
                    return
    if isinstance(readfile, dict):
        if 'image_file_list' in readfile.keys():
            if len(readfile['image_file_list']) > 0:
                readfile = readfile['image_file_list'][0]
            else:
                print('card_file include no file!')
                return

    # card_file = image_list[0] if (len(image_list[0]) > 0) & (len(file) == 0) else file
    if isinstance(readfile, list):
        if len(readfile) > 0:
            readfile = readfile[0]
        else:
            print('filelist is empty! please assign card_form or filename!')
            return
    if len(readfile) == 0:
        print('please assign card_form or filename!')
        return
    read4files = []
    if os.path.isdir(readfile):
        read4files = Util.glob_files_from_path(readfile, substr_list='')
        if len(read4files) > 0:
            readfile = read4files[0]
    if not os.path.isfile(readfile):
        print('%s does not exist!' % readfile)
        return

    # initiating form
    this_form = {
        'len': 1 if len(read4files) == 0 else len(read4files),
        'image_file_list': read4files if len(read4files) > 0 else [readfile],
        'omr_form_check_mark_from_bottom': True,
        'omr_form_check_mark_from_right': True,
        'mark_format': {
            'mark_col_number': 100,
            'mark_row_number': 100,
            'mark_valid_area_col_start': 1,
            'mark_valid_area_col_end': 10,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10,
            'mark_location_row_no': 50 if check_mark_frombottom else 1,
            'mark_location_col_no': 50 if check_mark_fromright else 1
        },
        'group_format': {},
        'image_clip': {
            'do_clip': False if clip_top + clip_bottom + clip_left + clip_right == 0 else True,
            'x_start': clip_left,
            'x_end': -1 if clip_right == 0 else -1 * clip_right,
            'y_start': clip_top,
            'y_end': -1 if clip_bottom == 0 else -1 * clip_bottom
        },
        'model_para': {
            'valid_painting_gray_threshold': 35,
            'valid_peak_min_width': 3,
            'valid_peak_min_max_width_ratio': 5,
            'detect_mark_vertical_window': detect_mark_vertical_window,
            'detect_mark_horizon_window': detect_mark_horizon_window,
            'detect_mark_step_length': detect_mark_step_length,
            'detect_mark_max_stepnum': detect_mark_max_stepnum
        },
        'image_clip2': {
            'do_clip': False if clip_top + clip_bottom + clip_left + clip_right == 0 else True,
            'clip_top': clip_top,
            'clip_bottom': clip_bottom,
            'clip_left': clip_left,
            'clip_right': clip_right
        }
    }
    omr = OmrModel()
    omr.set_form(this_form)
    omr.set_omr_image_filename(readfile)
    omr.sys_run_check = True
    omr.sys_display = True
    omr.check_max_stepnum = detect_mark_max_stepnum
    omr.check_mark_min_num = detect_mark_min_marknum
    omr.check_horizon_window = detect_mark_horizon_window
    omr.check_vertical_window = detect_mark_vertical_window
    omr.check_step_length = detect_mark_step_length
    # omr.omr_form_do_tilt_check = True

    # omr.run()
    # initiate some variables
    omr.pos_xy_start_end_list = [[], [], [], []]
    omr.pos_start_end_list_log = dict()
    omr.omr_result_dataframe = \
        pd.DataFrame({'card': [Util.find_path_from_pathfile(omr.image_filename).split('.')[0]],
                      'result': ['XXX'],
                      'len': [-1],
                      'group': [''],
                      'valid': [0]
                      }, index=[omr.card_index_no])
    omr.omr_result_dataframe_groupinfo = \
        pd.DataFrame({'coord': [(-1)],
                      'label': [-1],
                      'feat': [(-1)],
                      'group': [''],
                      'code': [''],
                      'mode': ['']
                      })
    # start running
    st_time = time.clock()
    omr.get_card_image(omr.image_filename)

    # detect mark area
    iter_count = detect_mark_max_stepnum
    steplen, stepwid = 5, 20
    leftmax, rightmax, topmax, bottommax = 0, 0, 0, 0
    for step in range(iter_count):
        if stepwid + step*steplen < omr.image_card_2dmatrix.shape[1]:
            cur_mean = omr.image_card_2dmatrix[:, step * steplen:stepwid + step * steplen].mean()
            leftmax = max(leftmax, cur_mean)
            cur_mean = omr.image_card_2dmatrix[:, -stepwid - step * steplen:-step * steplen-1].mean()
            rightmax = max(rightmax, cur_mean)
        if stepwid + step * steplen < omr.image_card_2dmatrix.shape[0]:
            cur_mean = omr.image_card_2dmatrix[step * steplen:stepwid + step * steplen, :].mean()
            topmax = max(topmax, cur_mean)
            # print('top mean=%4.2f' % cur_mean)
            cur_mean = omr.image_card_2dmatrix[-stepwid - step * steplen:-step * steplen-1, :].mean()
            bottommax = max(bottommax, cur_mean)
            # print('bottom mean=%4.2f' % cur_mean)
    print('-'*70)
    print('marginal gray level: left=%4.1f, right=%4.1f, top=%4.1f, bottom=%4.1f' %
          (leftmax, rightmax, topmax, bottommax))
    print('-'*70)
    check_mark_frombottom = True if bottommax > int(topmax * 0.8) else False
    check_mark_fromright = True if rightmax > int(leftmax * 0.8) else False
    omr.omr_form_check_mark_from_bottom = check_mark_frombottom
    omr.omr_form_check_mark_from_right = check_mark_fromright

    # detect mark position
    omr.get_mark_pos()  # for test, not create row col_start end_pos_list

    if (omr.pos_best_horizon_mark_count is None) or \
            (omr.pos_best_vertical_mark_count is None):
        print('cannot find valid map!')
        print('running consume %1.4f seconds' % (time.clock() - st_time))
        return omr   # , this_form

    test_col_number = len(omr.pos_start_end_list_log[('h', omr.pos_best_horizon_mark_count)][0])
    test_row_number = len(omr.pos_start_end_list_log[('v', omr.pos_best_vertical_mark_count)][0])

    valid_h_map = {k[1]: omr.pos_start_end_list_log[k] for k in omr.pos_start_end_list_log
                   if (len(omr.pos_start_end_list_log[k][0]) == test_col_number) & (k[0] == 'h')}
    valid_v_map = {k[1]: omr.pos_start_end_list_log[k] for k in omr.pos_start_end_list_log
                   if (len(omr.pos_start_end_list_log[k][0]) == test_row_number) & (k[0] == 'v')}
    valid_h_map_threshold = {k: omr.pos_prj_log[('h', k)].mean()*0.618 for k in valid_h_map}
    valid_v_map_threshold = {k: omr.pos_prj_log[('v', k)].mean()*0.618 for k in valid_v_map}

    print("-"*70+chr(10), 'check result:\n\t horizonal_mark_num =',
          '%3d' % test_col_number, '\n\t vertical_mark_num = %3d' % test_row_number)
    print('\t detect horizon  mark from  right:%s' % check_mark_fromright)
    print('\t detect vertical mark from bottom:%s' % check_mark_frombottom)

    print('-'*70 + '\ntesting with read_test() ...')

    # set some values in form
    this_form['mark_format']['mark_location_row_no'] = test_row_number if check_mark_frombottom else 1
    this_form['mark_format']['mark_location_col_no'] = test_col_number if check_mark_fromright else 1
    this_form['mark_format']['mark_row_number'] = test_row_number
    this_form['mark_format']['mark_col_number'] = test_col_number
    if check_mark_fromright:
        this_form['mark_format']['mark_valid_area_col_start'] = 1
        this_form['mark_format']['mark_valid_area_col_end'] = test_col_number - 1
    else:
        this_form['mark_format']['mark_valid_area_col_start'] = 2
        this_form['mark_format']['mark_valid_area_col_end'] = test_col_number
    if check_mark_frombottom:
        this_form['mark_format']['mark_valid_area_row_start'] = 1
        this_form['mark_format']['mark_valid_area_row_end'] = test_row_number - 1
    else:
        this_form['mark_format']['mark_valid_area_row_start'] = 2
        this_form['mark_format']['mark_valid_area_row_end'] = test_row_number
    this_form['omr_form_check_mark_from_bottom'] = check_mark_frombottom
    this_form['omr_form_check_mark_from_right'] = check_mark_fromright

    # get former
    this_former = __read_check_make_former(this_form)

    # run omr to indentify form parameter
    test_model = read_test(this_former)
    print(test_model.omr_result_dataframe)

    if not display_figures:
        print('running consume %1.4f seconds' % (time.clock() - st_time))
        R = namedtuple('result', ['check_model', 'test_model'])
        return R(omr, test_model)

    # display result
    fnum = __read_check_disp(1, 'h', omr, valid_h_map, valid_h_map_threshold)
    fnum = __read_check_disp(fnum+1, 'v', omr, valid_v_map, valid_v_map_threshold)
    plt.figure(fnum+1)
    omr.plot_image_raw_card()
    plt.figure(fnum+2)
    omr.plot_image_with_markline()
    plt.figure(fnum+3)
    test_model.plot_image_recogblocks()
    # plt.figure(fnum+4)
    test_model.plot_grid_with_blockpoints()

    # save form to xml or python_code
    if form2file != '':
        __read_check_saveform(form2file, readfile, this_form)

    print('-'*70)
    print('running consume %1.4f seconds' % (time.clock() - st_time))

    R = namedtuple('result', ['check_model', 'test_model'])
    return R(check_model=omr, test_model=test_model)


def __read_check_make_former(this_form):

    image_clip = this_form['image_clip2']
    file_list = this_form['image_file_list']
    mark_format = this_form['mark_format']
    model_para = this_form['model_para']
    # print(mark_format)

    # define former
    former = Former()

    # define model parameters
    former.set_model_para(
        valid_painting_gray_threshold=model_para['valid_painting_gray_threshold'],
        valid_peak_min_width=model_para['valid_peak_min_width'],
        valid_peak_min_max_width_ratio=model_para['valid_peak_min_max_width_ratio'],
        detect_mark_vertical_window=model_para['detect_mark_vertical_window'],
        detect_mark_horizon_window=model_para['detect_mark_horizon_window'],
        detect_mark_step_length=model_para['detect_mark_step_length'],
        detect_mark_max_stepnum=model_para['detect_mark_max_stepnum']
    )

    # define image clip setting
    former.set_clip(
        do_clip=image_clip['do_clip'],
        clip_left=image_clip['clip_left'],
        clip_right=image_clip['clip_right'],
        clip_top=image_clip['clip_top'],
        clip_bottom=image_clip['clip_bottom']
    )

    # define location for checking mark
    former.set_check_mark_from_bottom(this_form['omr_form_check_mark_from_bottom'])
    former.set_check_mark_from_right(this_form['omr_form_check_mark_from_right'])

    # define image files list
    former.file_list = file_list

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=mark_format['mark_row_number'],
        col_number=mark_format['mark_col_number'],
        valid_area_row_start=mark_format['mark_valid_area_row_start'],
        valid_area_row_end=mark_format['mark_valid_area_row_end'],
        valid_area_col_start=mark_format['mark_valid_area_col_start'],
        valid_area_col_end=mark_format['mark_valid_area_col_end'],
        location_row_no=mark_format['mark_location_row_no'],
        location_col_no=mark_format['mark_location_col_no']
    )

    # define cluster
    max_row_num = mark_format['mark_valid_area_row_end'] - \
        mark_format['mark_valid_area_row_start'] + 1
    max_col_num = mark_format['mark_valid_area_col_end'] - \
        mark_format['mark_valid_area_col_start'] + 1
    gl = []
    cl = []
    gno = 0
    for i in range(1, max_row_num+1):
        for j in range(1, max_col_num+1):
            gno = gno + 1
            gl.append((gno, gno))
            cl.append((i, j))
    former.set_cluster(
        cluster_group_list=gl,  # group scope (min_no, max_no) per area
        cluster_coord_list=cl,  # left_top coord per area
        area_direction='v',           # area direction V:top to bottom, H:left to right
        group_direction='h',          # group direction 'V','v': up to down, 'H','h': left to right
        group_code='1',               # group code for painting block
        group_mode='S'                # group mode 'M': multi_choice, 'S': single_choice, X: any char
    )
    return former


def __read_check_disp(fnum, hv, omr, valid_map, valid_map_threshold):
    # fnum = 1
    plt.figure(fnum)  # 'vertical mark check')
    disp = 1
    alldisp = 0
    for vcount in valid_map:
        plt.subplot(240+disp)
        plt.plot(omr.pos_prj_log[(hv, vcount)])
        plt.plot([valid_map_threshold[vcount]]*len(omr.pos_prj_log[(hv, vcount)]))
        plt.xlabel(hv+'_mapf ' + str(vcount))
        plt.subplot(244+disp)
        plt.plot(omr.pos_prj01_log[(hv, vcount)])
        plt.xlabel(hv+'_mark[' + str(vcount)+']  num=' +
                   str(omr.pos_start_end_list_log[(hv, vcount)][0].__len__()))
        alldisp += 1
        if alldisp == len(valid_map):
            break
        if disp == 4:
            fnum = fnum + 1
            plt.figure(fnum)
            disp = 1
        else:
            disp = disp + 1
    plt.show()
    return fnum


def __read_check_saveform(form2file, card_file, this_form):
    saveform = Former()
    stl = saveform.former_template.split('\n')
    stl = [s[8:] for s in stl]
    for n, s in enumerate(stl):
        if 'path=' in s:
            stl[n] = stl[n].replace("?", Util.find_path_from_pathfile(card_file))
        if 'substr_list=' in s:
                substr = ''
                if '.jpg' in card_file:
                    substr = '.jpg'
                stl[n] = stl[n].replace("$", Util.find_path_from_pathfile(substr))
        if 'row_number=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_row_number']))  # str(test_row_number))
        if 'col_number=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_col_number']))  # str(test_col_number))
        if 'valid_area_row_start=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_valid_area_row_start']))
        if 'valid_area_row_end=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_valid_area_row_end']))
        if 'valid_area_col_start=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_valid_area_col_start']))
        if 'valid_area_col_end=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_valid_area_col_end']))
        if 'location_row_no=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_location_row_no']))
        if 'location_col_no=' in s:
            stl[n] = stl[n].replace('?', str(this_form['mark_format']['mark_location_col_no']))
        if 'set_check_mark_from_bottom' in s:
            stl[n] = stl[n].replace('?', 'True' if this_form['omr_form_check_mark_from_bottom'] else 'False')
        if 'set_check_mark_from_right' in s:
            stl[n] = stl[n].replace('?', 'True' if this_form['omr_form_check_mark_from_right'] else 'False')
        if 'do_clip' in s:
            stl[n] = stl[n].replace('?', 'True' if this_form['image_clip']['do_clip'] else 'False')
        if 'clip_top' in s:
            stl[n] = stl[n].replace('?', str(this_form['image_clip']['y_start']))
        if 'clip_bottom' in s:
            clip = this_form['image_clip']['y_end']
            stl[n] = stl[n].replace('?', str(0 if clip == -1 else -1*clip))
        if 'clip_left' in s:
            stl[n] = stl[n].replace('?', str(this_form['image_clip']['x_start']))
        if 'clip_right' in s:
            clip = this_form['image_clip']['x_end']
            if clip < -1:
                stl[n] = stl[n].replace('?', str(-1 * clip))
            else:
                stl[n] = stl[n].replace('?', str(0))
        # score_dict = {1: {'A':1}, 2: {'B':1}}
        if 'score_d' in s:
            if 'score_d1' in s:
                stl[n] = stl[n].replace('score_d1?', "1: {'A': 1},")
            elif 'score_d2' in s:
                stl[n] = stl[n].replace('score_d2?', "2: {'A': 1},")
            elif 'score_d3' in s:
                stl[n] = stl[n].replace('score_d3?', "3: {'A': 1},")
            elif 'score_d4' in s:
                stl[n] = stl[n].replace('score_d4?', "4: {'A': 1},")
            elif 'score_d5' in s:
                stl[n] = stl[n].replace('score_d5?', "5: {'A': 1},")

    if os.path.isfile(form2file):
        fh = open(form2file, 'a')
        form_string = '\n' + '\n'.join(stl) + '\n'
    else:
        fh = open(form2file, 'w')
        form_string = '# _*_ utf-8 _*_\n\nimport omrlib\n\n' + \
                      '\n'.join(stl) + '\n'
    fh.write(form_string)
    fh.close()


class Coder(object):

    __doc__ = \
        '''
        code table for group = 'A, B, C， D' or 'A, B, C, D, E'
        code_type: gb, drs, n18, nhomr, bcd
        not   painting using '.'
        error painting using '>'
        n18 is extended from gb(ABCD) to (ABCDE)
        '''

    # NHOMR, multi choice from 'ABCDE'
    omr_code_dict_nhomr = \
        {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
         'F': 'BC', 'G': 'ABC', 'H': 'AB', 'I': 'AD',
         'J': 'BD', 'K': 'ABD', 'L': 'CD', 'M': 'ACD',
         'N': 'BCD', 'O': 'ABCD', 'P': 'AC', 'Q': 'AE',
         'R': 'BE', 'S': 'ABE', 'T': 'CE', 'U': 'ACE',
         'V': 'BCE', 'W': 'ABCE', 'X': 'DE', 'Y': 'ADE',
         'Z': 'BDE', '[': 'ABDE', '\\': 'CDE', ']': 'ACDE',
         '^': 'BCDE', '_': 'ABCDE',
         '.': '',  # no choice
         }

    # GB, multi choice from 'ABCD'
    omr_code_dict_gb = \
        {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
         'E': 'AB', 'F': 'AC', 'G': 'AD', 'H': 'BC', 'I': 'BD',
         'J': 'CD', 'K': 'ABC', 'L': 'ABD', 'M': 'ACD',
         'N': 'BCD', 'O': 'ABCD', 'P': ''
         }

    # DRS, multi choice from 'ABCD'
    omr_code_dict_drs = \
        {'A': 'A', 'B': 'B', 'C': 'AB', 'D': 'C',
         'E': 'AC', 'F': 'BC', 'G': 'ABC', 'H': 'D', 'I': 'AD',
         'J': 'BD', 'K': 'ABD', 'L': 'CD', 'M': 'ACD',
         'N': 'BCD', 'O': 'ABCD', '*': ''
         }

    # N18, multi choice from 'ABCDE'
    omr_code_dict_n18 = \
        {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
         '$': 'AB', 'F': 'AC', 'G': 'AD', 'H': 'BC', 'I': 'BD', 'J': 'CD',
         'K': 'ABC', 'L': 'ABD', 'M': 'ACD', 'N': 'BCD', 'O': 'ABCD', 'P': '',
         'Q': 'AE',  'R': 'BE', 'S': 'CE', 'T': 'DE',
         'U': 'ABE', 'V': 'ACE', 'W': 'ADE', 'X': 'BCE', 'Y': 'BDE', 'Z': 'CDE',
         '[': 'ABCE', ']': 'ABDE', '{': 'ACDE', '}': 'BCDE', '%': 'ABCDE'
         }

    # BCD, for 8421 mode
    omr_code_standard_dict_bcd = \
        {'1': '1', '2': '2', '3': '12', '4': '4', '5': '14',
         '6': '24', '7': '124', '8': '8', '9': '18', '0': ''}

    def __init__(self):
        self.code_tables_dict = {
            'gb': Coder.omr_code_dict_gb,
            'n18': Coder.omr_code_dict_n18,
            'drs': Coder.omr_code_dict_drs,
            'nhomr': Coder.omr_code_dict_nhomr,
            'bcd': Coder.omr_code_standard_dict_bcd
            }

    def add_code_talbe(self, code_type: str, code_dict):
        if code_type in self.code_tables_dict:
            print('warning: code type %s exists in coder dict!' % code_type)
        self.code_tables_dict.update({code_type: code_dict})

    def get_code_table(self, code_type):
        # return Coder.omr_code_standard_dict_nhomr
        if code_type in self.code_tables_dict:
            return self.code_tables_dict[code_type]
        else:
            print('invalid code type %s' % code_type)
            return dict()

    def get_encode_table(self, code_type):
        if code_type in self.code_tables_dict:
            ct = self.code_tables_dict[code_type]
            return {ct[k]: k for k in ct}
        else:
            print('invalid code type %s' % code_type)
            return dict()

    def code_switch(self, from_code_type, to_code_type, code_string):
        encode_dict = {self.code_tables_dict[to_code_type][k]: k for k in self.code_tables_dict[to_code_type]}
        new_code_string = ''
        for c in code_string:
            if sc in self.code_tables_dict[from_code_type]:
                sc = self.code_tables_dict[from_code_type][c]
            elif sc != '>':
                sc = '#'
                # print('no code %s in dict[%s], set to #!' % (sc, to_code_type))
            new_code_string = new_code_string + encode_dict[sc]
        return new_code_string


class Former:
    """
    card_form = {
        'image_file_list': omr_image_list,
        'iamge_clip':{
            'do_clip': False,
            'x_start': 0, 'x_end': 100, 'y_start': 0, 'y_end': 200
            },
        'omr_form_check_mark_from_bottome': True,
        'check_vertical_mark_from_top': True,
        'mark_format': {
            'mark_col_number': 37,
            'mark_row_number': 14,
            'mark_valid_area_col_start': 23,
            'mark_valid_area_col_end': 36,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 13,
            'mark_location_row_no':14,
            'mark_location_col_no':37
            },
        'group_format': {No: [(r,c),   # start position: (row number, column number)
                              int      # length
                              char     # direction, 'V'=vertical, 'H'=horizonal
                              str      # codestring,  for example: 'ABCD', '0123456789'
                              char     # choice mode, 'S'=single choice, 'M'=multi choice
                              ]}
        }
    of = OmrForm()
    of.set_imagefile(file_list:list)
    of.set_image_clip(x_satrt=0, x_end=-1, y_satrt=0, y_end=-1, do_clip=False)
    of.set_mark(col_number=37, row_number=14...)
    of.set_group(group_no=1, coord=(0,0), len=4, dir='H', code='ABCD', mode='S')
    of.check()   # check set error
    form=of.get_form()
    ------
    painting format:
    # : no block painted in a group
    > : invalid painting in a group (more than one block painted for single mode 'S')
    """

    _template = """
        def form_xxx():
            
            # define former
            former = omrlib.Former()
            
            # define model parameters
            former.set_model_para(
                valid_painting_gray_threshold=35,
                valid_peak_min_width=3,
                valid_peak_min_max_width_ratio=5,
                detect_mark_vertical_window=15,
                detect_mark_horizon_window=12,
                detect_mark_step_length=5,
                detect_mark_max_stepnum=20
                )
            
            # define image clip setting
            former.set_clip(
                do_clip=?,
                clip_left=?,
                clip_right=?,
                clip_top=?,
                clip_bottom=?
                )

            # define location for checking mark 
            former.set_check_mark_from_bottom(?)
            former.set_check_mark_from_right(?)
            
            # define image files list
            former.set_file_list(
                path='?', 
                substr_list='jpg'    # assign substr in path to filter
                )
            
            # define mark format: row/column number, valid area, location
            former.set_mark_format(
                row_number=?,
                col_number=?,
                valid_area_row_start=?,
                valid_area_row_end=?,
                valid_area_col_start=?,
                valid_area_col_end=?,
                location_row_no=?,
                location_col_no=?
                )
            
            # define code area, containing many groups
            former.set_area(
                area_group=(1, 15),         # area group from min=a to max=b (a, b)
                area_coord=(10, 20),        # area location left_top = (row, col)
                area_direction='h',         # area direction V:top to bottom, H:left to right
                group_direction='v',        # group direction from left to right
                group_code='0123456789',    # group code for painting block
                group_mode='D'              # group mode 'M': multi_choice, 'S': single_choice, 'D':digit
                )
            
            # define cluster
            former.set_cluster(
                cluster_group_list=[(100, 105), (106, 110)],    # group scope (min_no, max_no) per area
                cluster_coord_list=[(1, 3), (1, 9)],            # left_top coord per area
                area_direction='v',      # area direction V:top to bottom, H:left to right
                group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
                group_code='ABCD',       # group code for painting block
                group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
                )
            
            # define score_dict
            # {group:{code:score,}}
            former.set_score(
                do_score=False,
                score_dict={
                    score_d1?
                    score_d2?
                    score_d3?
                    score_d4?
                    score_d5?
                }
            )
            
            return former"""

    def __init__(self):
        self.former_template = Former._template
        self.form = dict()
        self.file_list = list()
        self.mark_format = dict()
        self.group_format = dict()
        self.model_para = {
            'valid_painting_gray_threshold': 35,
            'valid_peak_min_width': 3,
            'valid_peak_min_max_width_ratio': 5,
            'detect_mark_vertical_window': 15,
            'detect_mark_horizon_window': 12,
            'detect_mark_step_length': 5,
            'detect_mark_max_stepnum': 20
        }
        self.image_clip = {
            'do_clip': False,
            'x_start': 0,
            'x_end': -1,
            'y_start': 0,
            'y_end': -1}
        self.omr_form_check_mark_from_bottom = True
        self.omr_form_check_mark_from_right = True
        self.score_dict = dict()
        self.do_score = False

    @classmethod
    def help(cls):
        print(cls.__doc__)

    def set_file_list(self, path: str, substr_list):
        self.file_list = Util.glob_files_from_path(path, substr_list)
        self._make_form()

    def set_model_para(
            self,
            valid_painting_gray_threshold=35,
            valid_peak_min_width=3,
            valid_peak_min_max_width_ratio=5,
            detect_mark_vertical_window=15,
            detect_mark_horizon_window=12,
            detect_mark_step_length=5,
            detect_mark_max_stepnum=20
            ):
        self.model_para = {
            'valid_painting_gray_threshold': valid_painting_gray_threshold,
            'valid_peak_min_width': valid_peak_min_width,
            'valid_peak_min_max_width_ratio': valid_peak_min_max_width_ratio,
            'detect_mark_vertical_window': detect_mark_vertical_window,
            'detect_mark_horizon_window': detect_mark_horizon_window,
            'detect_mark_step_length': detect_mark_step_length,
            'detect_mark_max_stepnum': detect_mark_max_stepnum
        }

    def set_image_clip(
            self,
            clip_x_start=1,
            clip_x_end=-1,
            clip_y_start=1,
            clip_y_end=-1,
            do_clip=False):
        self.image_clip = {
            'do_clip': do_clip,
            'x_start': clip_x_start,
            'x_end': clip_x_end,
            'y_start': clip_y_start,
            'y_end': clip_y_end
        }
        self._make_form()

    def set_clip(
            self,
            do_clip=False,
            clip_left=0,
            clip_right=0,
            clip_top=0,
            clip_bottom=0
            ):
        self.image_clip = {
            'do_clip': do_clip,
            'x_start': clip_left,
            'x_end': -1 if clip_right == 0 else -1 * clip_right,
            'y_start': clip_top,
            'y_end': -1 if clip_bottom == 0 else -1 * clip_bottom
        }
        self._make_form()

    def set_check_mark_from_bottom(self, mode=True):
        self.omr_form_check_mark_from_bottom = mode
        self._make_form()

    def set_check_mark_from_right(self, mode=True):
        self.omr_form_check_mark_from_right = mode
        self._make_form()

    def set_mark_format(
            self,
            col_number: int,
            row_number: int,
            valid_area_col_start: int,
            valid_area_col_end: int,
            valid_area_row_start: int,
            valid_area_row_end: int,
            location_row_no: int,
            location_col_no: int
            ):
        self.mark_format = {
            'mark_col_number': col_number,
            'mark_row_number': row_number,
            'mark_valid_area_col_start': valid_area_col_start,
            'mark_valid_area_col_end': valid_area_col_end,
            'mark_valid_area_row_start': valid_area_row_start,
            'mark_valid_area_row_end': valid_area_row_end,
            'mark_location_row_no': location_row_no,
            'mark_location_col_no': location_col_no
        }
        self._make_form()

    def set_group_coord(self, group_no, group_coord):
        if group_no in self.group_format:
            oldgroup = self.group_format[group_no]
            self.group_format.update({
                group_no: [group_coord, oldgroup[1], oldgroup[2], oldgroup[3], oldgroup[4]]
            })
            self._make_form()
        else:
            print('invalid group no = %s' % group_no)

    def set_group_direction(self, group_no, group_dire):
        if group_no in self.group_format:
            oldgroup = self.group_format[group_no]
            self.group_format.update({
                group_no: [oldgroup[0], oldgroup[1], group_dire, oldgroup[3], oldgroup[4]]
            })
            self._make_form()
        else:
            print('invalid group no = %s' % group_no)

    def set_group_code(self, group_no, group_code):
        if group_no in self.group_format:
            oldgroup = self.group_format[group_no]
            self.group_format.update({
                group_no: [oldgroup[0], oldgroup[1], oldgroup[2], group_code, oldgroup[4]]
            })
            self._make_form()
        else:
            print('invalid group no = %s' % group_no)

    def set_group_mode(self, group_no, group_mode):
        if group_no in self.group_format:
            oldgroup = self.group_format[group_no]
            self.group_format.update({
                group_no: [oldgroup[0], len(oldgroup[3]), oldgroup[2], oldgroup[3], group_mode]
            })
            self._make_form()
        else:
            print('invalid group no = %s' % group_no)

    def set_group(self, group: int, coord: tuple, group_direction: str, group_code: str, group_mode: str):
        self.group_format.update({
            group: [coord, len(group_code), group_direction.upper(), group_code, group_mode]
        })
        self._make_form()

    def set_area(self,
                 area_group: (int, int),
                 area_coord: (int, int),
                 area_direction='v',
                 group_direction='V',
                 group_code='ABCD',
                 group_mode='S'
                 ):
        area_h_move = 1 if area_direction.upper() == 'H' else 0
        area_v_move = 1 if area_direction.upper() == 'V' else 0
        for gn in range(area_group[0], area_group[1]+1):
            self.set_group(group=gn,
                           coord=(area_coord[0] + area_v_move * (gn - area_group[0]),
                                  area_coord[1] + area_h_move * (gn - area_group[0])),
                           group_direction=group_direction,
                           group_code=group_code,
                           group_mode=group_mode
                           )
        # _make_form in set_group

    def set_cluster(self,
                    cluster_group_list,      # group scope (min_no, max_no) for each related area
                    cluster_coord_list,      # left_top coord for each related area
                    area_direction='v',      # area direction V:top to bottom, H:left to right
                    group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
                    group_code='ABCD',       # group code for painting block
                    group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
                    ):
        for gno, loc in zip(cluster_group_list, cluster_coord_list):
            self.set_area(
                area_group=gno,  # area group from min=a to max=b (a, b)
                area_coord=loc,  # area location left_top = (row, col)
                area_direction=area_direction,  # area direction V:top to bottom, H:left to right
                group_direction=group_direction,  # group direction 'V','v': up to down, 'H','h': left to right
                group_code=group_code,  # group code for painting block
                group_mode=group_mode  # group mode 'M': multi_choice, 'S': single_choice
            )
        # _make_form in set_area

    def set_score(self, do_score=None, score_dict=None):
        if do_score is not None:
            self.do_score = do_score
        if score_dict is not None:
            self.score_dict = score_dict
        self._make_form()

    def _make_form(self):
        if len(self.file_list) == 0:
            file_list = []
        elif (len(self.file_list) == 1) & (len(self.file_list[0]) == 0):
            file_list = []
        else:
            file_list = self.file_list
        self.form = {
            'image_file_list': file_list,
            'image_clip': self.image_clip,
            'mark_format': self.mark_format,
            'group_format': self.group_format,
            'omr_form_check_mark_from_bottom': self.omr_form_check_mark_from_bottom,
            'omr_form_check_mark_from_right': self.omr_form_check_mark_from_right,
            'model_para': self.model_para,
            'score_format': {'score_dict': self.score_dict,
                             'do_score': self.do_score}
        }
        # return self.form

    def _check_mark(self):
        # self.get_form()
        if len(self.form['image_file_list']) > 0:
            image_file = self.form['image_file_list'][0]
        else:
            # print('no file in image_file_list!')
            return
        # card_image = mg.imread(image_file)
        card_image = plt.imread(image_file)
        image_rawcard = card_image
        if self.form['image_clip']['do_clip']:
            card_image = image_rawcard[
                                  self.form['image_clip']['y_start']:self.form['image_clip']['y_end'],
                                  self.form['image_clip']['x_start']:self.form['image_clip']['x_end']]
        # image: 3d to 2d
        if len(card_image.shape) == 3:
            card_image = card_image.mean(axis=2)
        sh0, sh1 = card_image.shape[0], card_image.shape[1]
        # mark color is black
        card_image = 255 - card_image

        # set mark location # moving window to detect mark area
        steplen, stepwid = 5, 12
        leftmax, rightmax, topmax, bottommax = 0, 0, 0, 0
        for step in range(30):
            if stepwid + step*steplen < sh1:  # card_image.shape[1]:
                leftmax = max(leftmax, card_image[:, step * steplen:stepwid + step * steplen].mean())
                rightmax = max(rightmax, card_image[:, -stepwid - step * steplen:-step * steplen-1].mean())
            if stepwid + step*steplen < sh0:  # card_image.shape[0]:
                topmax = max(topmax, card_image[step * steplen:stepwid + step * steplen, :].mean())
                bottommax = max(bottommax, card_image[-stepwid - step * steplen:-step * steplen-1, :].mean())
        print('check vertical mark from  right: ', leftmax < rightmax)
        print('check horizon  mark from bottom: ', topmax < bottommax)
        self.omr_form_check_mark_from_bottom = True if topmax < bottommax else False
        self.omr_form_check_mark_from_right = True if rightmax > leftmax else False
        self._make_form()

    def show_form(self):
        # show format
        for k in self.form.keys():
            if k == 'group_format':
                print('group_format: {0} ... {1}'.
                      format(list(self.form[k].values())[0],
                             list(self.form[k].values())[-1])
                      )
            elif k == 'mark_format':
                # print('mark_formt:')
                print(' mark_format: row={0}, col={1};  valid_row=[{2}-{3}], valid_col=[{4}-{5}];  '.
                      format(
                        self.form['mark_format']['mark_row_number'],
                        self.form['mark_format']['mark_col_number'],
                        self.form['mark_format']['mark_valid_area_row_start'],
                        self.form['mark_format']['mark_valid_area_row_end'],
                        self.form['mark_format']['mark_valid_area_col_start'],
                        self.form['mark_format']['mark_valid_area_col_end'])
                      + 'location_row={0}, location_col={1};'.
                      format(
                             self.form['mark_format']['mark_location_row_no'],
                             self.form['mark_format']['mark_location_col_no'])
                      )
            elif k == 'model_para':
                # model_para_str = '{' + \
                #    '\n\t\t\tgray_threshold:' + str(self.form['model_para']['valid_painting_gray_threshold']) + \
                #    '\n\t\t\tmin_peak_width:' + str(self.form['model_para']['valid_peak_min_width']) + \
                #    '\n\t\t\tpeak_wid_ratio:' + str(self.form['model_para']['valid_peak_min_max_width_ratio']) + \
                #    '\n\t\t\thorizon_window:' + str(self.form['model_para']['detect_mark_horizon_window']) + \
                #    '\n\t\t\tvertica_window:' + str(self.form['model_para']['detect_mark_vertical_window']) + \
                #    '\n\t\t\t   step_length:' + str(self.form['model_para']['detect_mark_step_length']) + \
                #    '\n\t\t\t   max_stepnum:' + str(self.form['model_para']['detect_mark_max_stepnum']) + \
                #    '\n\t\t\t}'
                print('  model_para:', self.form['model_para'])
                continue
            elif k == 'image_file_list':
                continue
            elif k == 'omr_form_check_mark_from_bottom':
                print('  check_mark: {0}, {1}'.
                      format('from bottom' if self.form[k] else 'from top',
                             'from right' if self.form[k] else 'from left'))
            elif k == 'omr_form_check_mark_from_right':
                continue
            elif k == 'image_clip':
                print('  image_clip: do_clip=', self.form[k]['do_clip'],
                      ' clip_top=', self.form[k]['y_start'],
                      ' clip_bottom=', -self.form[k]['y_end'] if self.form[k]['y_end'] < 0
                      else self.form[k]['y_end'],
                      ' clip_left=', self.form[k]['x_start'],
                      ' clip_right=', -self.form[k]['x_end'] if self.form[k]['x_end'] < 0
                      else self.form[k]['x_end'])
            else:
                print(k+':', self.form[k])
        # show files retrieved from assigned_path
        if 'image_file_list' in self.form.keys():
            if len(self.form['image_file_list']) > 0:
                print('   file_list:',
                      self.form['image_file_list'][0],
                      '...  files_number= ', len(self.form['image_file_list']))
            else:
                print('image_file_list: empty!')

    def show_group(self):
        pp.pprint(self.form['group_format'])

    def show_image(self, index=0):
        if self.form['image_file_list'].__len__() > 0:
            if index in range(self.form['image_file_list'].__len__()):
                f0 = self.form['image_file_list'][index]
            else:
                print('index is no in range(file_list_len)!')
                return
            if os.path.isfile(f0):
                im0 = mg.imread(f0)
                plt.figure(10)
                plt.imshow(im0)
                plt.title(f0)
            else:
                print('invalid file in form')
        else:
            print('no file in form')


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
        self.card_index_no = 0
        self.image_filename = ''
        self.image_rawcard = None
        self.image_card_2dmatrix = None  # np.zeros([3, 3])
        self.image_blackground_with_rawblock = None
        self.image_blackground_with_recogblock = None
        self.omr_kmeans_cluster = KMeans(2)
        # self.cnnmodel = OmrCnnModel()
        # self.cnnmodel.load_model('m18test')     # trained by 20000 * 40batch to accuracy==1.0

        # omr form parameters
        self.form = dict()
        self.omr_form_mark_area = {'mark_horizon_number': 20, 'mark_vertical_number': 11}
        self.omr_form_valid_area = {'mark_horizon_number': [1, 19], 'mark_vertical_number': [1, 10]}
        self.omr_form_group_dict = {1: [(0, 0), 4, 'H', 'ABCD', 'S']}  # pos, len, dir, code, mode
        self.omr_form_coord_group_dict = {}
        self.omr_form_image_do_clip = False
        self.omr_form_image_clip_area = []
        self.omr_form_do_tilt_check = False
        self.omr_form_mark_location_row_no = 0
        self.omr_form_mark_location_col_no = 0
        self.omr_form_check_mark_from_right = True
        self.omr_form_check_mark_from_bottom = True

        # system control parameters
        self.sys_run_test = False
        self.sys_run_check = False
        self.sys_group_result = False
        self.sys_display = False        # display time, error messages in running process
        self.sys_logwrite = False       # record processing messages in log file, finished later

        # model parameter
        self.check_gray_threshold = 35
        self.check_peak_min_width = 4
        self.check_mark_min_num = 3     # min mark number in row and column
        self.check_peak_min_max_width_ratio = 5
        self.check_mark_gap_top_std = 50
        self.check_mark_mapf_low_std = 30
        self.check_mark_peak_top_var = 100
        self.check_block_min_gray_mean = 12.5   # visible level as painting block
        self.check_vertical_window: int = 15
        self.check_horizon_window: int = 12
        self.check_step_length: int = 5
        self.check_max_stepnum = 20
        self.check_block_by_floating = False
        self.check_block_x_extend = 2
        self.check_block_y_extend = 1

        # check position data
        self.pos_x_prj_list: list = []
        self.pos_y_prj_list: list = []
        self.pos_xy_start_end_list: list = [[], [], [], []]
        self.pos_prj_log = dict()
        self.pos_prj01_log = dict()
        self.pos_valid_hmapfun_std_log = dict()
        self.pos_valid_vmapfun_std_log = dict()
        self.pos_peak_wid_var_log = dict()
        self.pos_start_end_list_log = dict()
        self.pos_best_horizon_mark_count = None
        self.pos_best_vertical_mark_count = None

        # recog result data
        self.omr_result_coord_blockimage_dict = {}
        self.omr_result_coord_markimage_dict = {}
        self.omr_result_horizon_tilt_rate = []   # [0 for _ in self.omr_form_mark_area['mark_horizon_number']]
        self.omr_result_vertical_tilt_rate = []  # [0 for _ in self.omr_form_mark_area['mark_vertical_number']]
        self.omr_result_data_dict = {}
        self.omr_result_dataframe = None
        self.omr_result_dataframe_groupinfo = None
        self.omr_result_save_blockimage_path = ''

        # omr encoding dict
        self.coder = Coder()
        # self.omr_bcd_code = self.coder.get_code_table('bcd')
        self.omr_bcd_encode = self.coder.get_encode_table('bcd')
        self.omr_code_type = 'n18'
        self.omr_encode_dict = self.coder.get_encode_table(self.omr_code_type)

    def run(self):
        # initiate some variables
        self.pos_xy_start_end_list = [[], [], [], []]
        if self.sys_run_test or self.sys_run_check:
            self.pos_start_end_list_log = dict()
            self.pos_prj_log = dict()
            self.pos_prj01_log = dict()
            self.pos_peak_wid_var_log = dict()
        self.omr_result_horizon_tilt_rate = \
            np.array([0 for _ in range(self.omr_form_mark_area['mark_horizon_number'])])
        self.omr_result_vertical_tilt_rate = \
            np.array([0 for _ in range(self.omr_form_mark_area['mark_vertical_number'])])

        # start running
        st = time.clock()
        # --get_image, get_pos, get_tilt, get_block, get_data, get_frame
        self.get_card_image(self.image_filename)
        if self.get_mark_pos():     # create row col_start end_pos_list
            if self.omr_form_do_tilt_check:  # check tilt
                self._check_mark_tilt()
            self._get_coord_blockimage_dict()
            self._get_result_data_dict()
            self._get_result_dataframe()
        else:
            self._set_result_dataframe_default()

        if self.sys_display:
            print('running consume %1.4f seconds' % (time.clock()-st))

    def set_code_table(self, code_type):
        if code_type in self.coder.code_tables_dict:
            self.omr_code_type = code_type
            self.omr_encode_dict = self.coder.get_encode_table(code_type)
            return True
        if self.sys_display:
            print('set code table fail! invalid code type = %s' % code_type)
        return False

    def set_form(self, card_form):

        # set form
        self.form = card_form

        # set mark_format
        # mark_format = [v for v in card_form['mark_format'].values()]
        self.set_mark_format(card_form)

        # set group
        # group = card_form['group_format']
        self.set_group(card_form['group_format'])

        # sel clip
        self.omr_form_image_do_clip = card_form['image_clip']['do_clip']
        # area_xend = card_form['image_clip']['x_end']
        # area_yend = card_form['image_clip']['y_end']
        self.omr_form_image_clip_area = [
            card_form['image_clip']['x_start'],
            card_form['image_clip']['x_end'],
            card_form['image_clip']['y_start'],
            card_form['image_clip']['y_end']]
        # set check from
        if 'omr_form_check_mark_from_bottom' in card_form.keys():
            self.omr_form_check_mark_from_bottom = card_form['omr_form_check_mark_from_bottom']
        if 'omr_form_check_mark_from_right' in card_form.keys():
            self.omr_form_check_mark_from_right = card_form['omr_form_check_mark_from_right']

        # set model para
        if 'model_para' in card_form.keys():
            self.check_gray_threshold = \
                card_form['model_para']['valid_painting_gray_threshold']
            self.check_peak_min_width = \
                card_form['model_para']['valid_peak_min_width']
            self.check_peak_min_max_width_ratio = \
                card_form['model_para']['valid_peak_min_max_width_ratio']
            self.check_max_stepnum = \
                card_form['model_para']['detect_mark_max_stepnum']
            self.check_step_length = \
                card_form['model_para']['detect_mark_step_length']
            self.check_horizon_window = \
                card_form['model_para']['detect_mark_horizon_window']
            self.check_vertical_window = \
                card_form['model_para']['detect_mark_vertical_window']
        else:
            if self.sys_display:
                print('--use default model parameters!')

    def set_mark_format(self, form):
        # set mark_location and check_tilt
        if ('mark_location_row_no' in form['mark_format'].keys()) & \
                ('mark_location_col_no' in form['mark_format'].keys()):
            self.omr_form_mark_location_row_no = form['mark_format']['mark_location_row_no']
            self.omr_form_mark_location_col_no = form['mark_format']['mark_location_col_no']
            self.omr_form_do_tilt_check = True
        else:
            self.omr_form_do_tilt_check = False
            if self.sys_display:
                print('no mark_location, set tilt_check fail!')
        # set mark_format
        if ('mark_row_number' in form['mark_format'].keys()) & \
                ('mark_col_number' in form['mark_format'].keys()) & \
                ('mark_valid_area_row_start' in form['mark_format'].keys()) & \
                ('mark_valid_area_row_end' in form['mark_format'].keys()) & \
                ('mark_valid_area_col_start' in form['mark_format'].keys()) & \
                ('mark_valid_area_col_end' in form['mark_format'].keys()):
            self.omr_form_mark_area['mark_horizon_number'] = form['mark_format']['mark_col_number']
            self.omr_form_mark_area['mark_vertical_number'] = form['mark_format']['mark_row_number']
            self.omr_form_valid_area['mark_horizon_number'] = [form['mark_format']['mark_valid_area_col_start'],
                                                               form['mark_format']['mark_valid_area_col_end']]
            self.omr_form_valid_area['mark_vertical_number'] = [form['mark_format']['mark_valid_area_row_start'],
                                                                form['mark_format']['mark_valid_area_row_end']]
        else:
            if self.sys_display:
                print('mark format keys is loss, set_mark_format fail!')

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
        if type(group) != dict:
            print('error: group_format is not a dict!')
            return
        self.omr_form_group_dict = group
        # self.omr_code_valid_number = 0
        for gno in group.keys():
            if (type(group[gno][0]) not in [tuple, list]) | \
                    (len(group[gno][0]) != 2):
                print('error: group-pos, group_format[0] is nor tuple like (r, c)!')
                return
            if len(group[gno]) != 5:
                print('error: group_format is not tuple length=5 !')
                return
            if type(group[gno][1]) != int:
                print('error: group-len, group_format[1]\'s type is not int!')
                return
            if type(group[gno][2]) != str:
                print('error: group-code, group_format[2]\'s type is not str!')
                return
            if type(group[gno][3]) != str:
                print('error: group-mode, group_format[3]\'s type is not str!')
                return
            # get pos coordination (row, col)
            r, c = self.omr_form_group_dict[gno][0]
            for j in range(self.omr_form_group_dict[gno][1]):
                # add -1 to set to 0 ... n-1 mode
                rt, ct = (r-1+j, c-1) if self.omr_form_group_dict[gno][2] in ['V', 'v'] else (r-1, c-1+j)
                # create (x, y):[gno, code, mode]
                self.omr_form_coord_group_dict[(rt, ct)] = \
                    (gno, self.omr_form_group_dict[gno][3][j], self.omr_form_group_dict[gno][4])
                # check (r, c) in mark area
                hscope = self.omr_form_valid_area['mark_horizon_number']
                vscope = self.omr_form_valid_area['mark_vertical_number']
                if (ct not in range(hscope[1])) | (rt not in range(vscope[1])):
                    print(f'group set error: ({rt+1}, {ct+1}) not in valid mark area{vscope}, {hscope}!')

    def set_omr_image_filename(self, file_name: str):
        self.image_filename = file_name

    def get_card_image(self, image_file):
        self.image_rawcard = mg.imread(image_file)
        self.image_card_2dmatrix = self.image_rawcard
        if self.omr_form_image_do_clip:
            self.image_card_2dmatrix = self.image_rawcard[
                                       self.omr_form_image_clip_area[2]:self.omr_form_image_clip_area[3],
                                       self.omr_form_image_clip_area[0]:self.omr_form_image_clip_area[1]]
        self.image_card_2dmatrix = 255 - self.image_card_2dmatrix
        # image: 3d to 2d
        if len(self.image_card_2dmatrix.shape) == 3:
            self.image_card_2dmatrix = self.image_card_2dmatrix.mean(axis=2)

    def save_result_omriamge(self):
        if self.omr_result_save_blockimage_path == '':
            print('to set save data path!')
            return
        if not os.path.exists(self.omr_result_save_blockimage_path):
            print(f'save data path "{self.omr_result_save_blockimage_path}" not exist!')
            return
        for coord in self.omr_result_coord_blockimage_dict:
            f = self.omr_result_save_blockimage_path + '/omr_block_' + str(coord) + '_' + \
                Util.find_file_from_pathfile(self.image_filename)
            mg.imsave(f, self.omr_result_coord_blockimage_dict[coord])

    def get_mark_pos(self):

        # check horizonal mark blocks (columns number)
        r1, steplen, stepcount = self._check_mark_seek_pos(self.image_card_2dmatrix,
                                                           mark_is_horizon=True,
                                                           window=self.check_horizon_window)
        if (stepcount < 0) & (not self.sys_run_check):
            return False

        # check vertical mark blocks (rows number)
        # if row marks check succeed, use row mark bottom zone to create map-fun for removing noise
        # rownum = self.image_card_2dmatrix.shape[0]
        # rownum = rownum - steplen * stepcount + 10  # remain gap for tilt, avoid to cut mark_edge
        r2, steplen, stepcount = self._check_mark_seek_pos(self.image_card_2dmatrix,  # [0:rownum, :],
                                                           mark_is_horizon=False,
                                                           window=self.check_vertical_window)
        if stepcount >= 0:
            if (len(r1[0]) > 0) | (len(r2[0]) > 0):
                self.pos_xy_start_end_list = np.array([r1[0], r1[1], r2[0], r2[1]])
                return True
        else:
            return False

    def _check_mark_seek_pos(self, img, mark_is_horizon, window):

        # _check_time = time.time()

        # dynamical step
        steplen = self.check_step_length
        win = self.check_horizon_window if mark_is_horizon else self.check_vertical_window

        # choose best mapfun with optimizing 0.6widths_var + 0.4gap_var
        if mark_is_horizon:
            self.pos_valid_hmapfun_std_log = dict()
        else:
            self.pos_valid_vmapfun_std_log = dict()
        # self.pos_peak_wid_var_log = dict()

        mark_start_end_position_dict = dict()
        mark_save_num = 0
        mark_save_max = 3

        mark_direction = 'horizon' if mark_is_horizon else 'vertical'
        dire = 'h' if mark_is_horizon else 'v'
        check_location = self.omr_form_check_mark_from_bottom \
            if mark_is_horizon else \
            self.omr_form_check_mark_from_right

        maxlen = self.image_card_2dmatrix.shape[0] \
            if mark_is_horizon else self.image_card_2dmatrix.shape[1]

        stepcount = 0
        cur_look = 0
        while True:

            cur_look = cur_look + steplen
            stepcount += 1

            # control check start location
            if check_location:
                start_line = maxlen - win - cur_look
                end_line = maxlen - cur_look
            else:
                start_line = cur_look
                end_line = win + cur_look

            # no mark area found
            if (maxlen < win + steplen * stepcount) | (stepcount > self.check_max_stepnum):
                if self.sys_display:
                    if not (self.sys_run_test or self.sys_run_check):
                        print('check mark fail: %s, step=%3d, steplen=%3d' %
                              (mark_direction, stepcount, steplen),
                              'detect_win=%3d, zone= [%4d:%4d]' % (window, start_line, end_line))
                    else:
                        print('check mark stop: %s, step=%3d, steplen=%3d' %
                              (mark_direction, stepcount, steplen),
                              'detect_win=%3d, zone= [%4d:%4d]' % (window, start_line, end_line))
                break

            map_fun = img[start_line:end_line, :].sum(axis=0) \
                if mark_is_horizon else \
                img[:, start_line:end_line].sum(axis=1)
            # map_fun = list(map_fun)
            # print('x0 step={0}, consume_time={1}'.format(stepcount, time.time()-_check_time))
            if self.sys_run_test or self.sys_run_check:
                self.pos_prj_log.update({(dire, stepcount): map_fun.copy()})

            # remove too small var for mapfun, no enough info to create mark peaks
            map_std = np.std(map_fun)
            valley = Util.seek_valley_wid_from_mapfun(map_fun)
            map_gap_std = np.std(valley) if len(valley) > 0 else 0
            # too_small_var to consume too much time in cluster, or no enough information in mapfun to cluster
            if map_std <= self.check_mark_mapf_low_std:
                if self.sys_display:
                    ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                         'num=%3d, map_std(%3.2f) is too small!'
                    print(ps % (mark_direction, stepcount, steplen, start_line, end_line, 0, map_std))
                continue
            # too large means a non-uniform distribution for gaps-wid, or a singular gaps-wid
            if map_gap_std > self.check_mark_gap_top_std:
                if self.sys_display:
                    ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                         'num=%3d, gap_std(%3.2f) is too large!'
                    print(ps % (mark_direction, stepcount, steplen, start_line, end_line, 0, map_gap_std))
                continue

            # get start-end pos list, smooth sharp-peak & sharp-valley in _byconv
            mark_start_end_pos_list, prj01 = self._check_mark_pos_byconv(map_fun)  # , mark_is_horizon)

            # record poslist and mapfun
            if self.sys_run_test or self.sys_run_check:
                self.pos_start_end_list_log.update({(dire, stepcount): mark_start_end_pos_list})
                self.pos_prj01_log.update({(dire, stepcount): prj01})

            # check mark number
            mark_num = len(mark_start_end_pos_list[0])
            if mark_num < self.check_mark_min_num:
                if self.sys_display:
                    ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                         'num=%3d, mark_num is too little!'
                    print(ps % (mark_direction, stepcount, steplen, start_line, end_line, mark_num))
                continue
            if not self.sys_run_check:
                form_mark_num = self.omr_form_mark_area['mark_horizon_number'] if mark_is_horizon else \
                                self.omr_form_mark_area['mark_vertical_number']
                if mark_num != form_mark_num:
                    if self.sys_display:
                        ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                             'check_num(%2d) != form_num(%2d)'
                        print(ps % (mark_direction, stepcount, steplen, start_line, end_line,
                              mark_num, form_mark_num))
                    continue

            # save valid mark_result
            if self._check_mark_pos_evaluate(mark_is_horizon,
                                             mark_start_end_pos_list,
                                             stepcount, start_line, end_line, steplen):
                if self.sys_display:
                    ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                         'num=%3d, map_std=%4.2f, gap_std=%4.2f'
                    print(ps % (mark_direction, stepcount, steplen, start_line, end_line,
                          mark_num, map_std, map_gap_std))
                mark_start_end_position_dict.update({stepcount: mark_start_end_pos_list})
                if mark_is_horizon:
                    self.pos_valid_hmapfun_std_log.update({stepcount: (map_std, map_gap_std)})
                else:
                    self.pos_valid_vmapfun_std_log.update({stepcount: (map_std, map_gap_std)})
                mark_save_num = mark_save_num + 1
                if not (self.sys_run_test or self.sys_run_check):
                    self.pos_prj01_log.update({(dire, stepcount): prj01})

            if not self.sys_run_check:
                # efficient valid mark number
                if mark_save_num == mark_save_max:
                    break

                # dynamical steplen
                if mark_save_num > 0:
                    steplen = 3
        # end while

        if self.sys_display:
            if mark_save_num == 0:
                if self.sys_run_check:
                    print('--check %s mark end--!' % mark_direction)
                else:
                    print('--check %s mark fail--!' % mark_direction)

        if mark_save_num > 0:
            # opt_count = self._check_mark_sel_opt(mark_start_end_position_dict)
            opt_count = self._check_mark_sel_opt2(mark_is_horizon)
            if mark_direction == 'horizon':
                self.pos_best_horizon_mark_count = opt_count
                self.pos_x_prj_list = self.pos_prj01_log[('h', opt_count)]
            else:
                self.pos_best_vertical_mark_count = opt_count
                self.pos_y_prj_list = self.pos_prj01_log[('v', opt_count)]
            if opt_count is not None:
                if self.sys_display:
                    print('--best step={0} in {1}'.format(opt_count, mark_start_end_position_dict.keys()))
                return mark_start_end_position_dict[opt_count], steplen, opt_count

        return [[], []], steplen, -1

    def _check_mark_pos_evaluate(self, horizon_mark, poslist, stepcount, start_line, end_line, steplen):

        hvs = 'horizon' if horizon_mark else 'vertical'

        # start position number is not same with end posistion number
        if len(poslist[0]) != len(poslist[1]):
            if self.sys_display:
                ps = 'detect mark %s, step=%3d, steplen=%3d, ' + \
                     'zone=[%4d--%4d] start_num(%2d != end_num(%2d)'
                print(ps % (hvs, stepcount, steplen, start_line, end_line, len(poslist[0]), len(poslist[1])))
            return False

        # pos error: start pos less than end pos
        tl = np.array([x2-x1+1 for x1, x2 in zip(poslist[0], poslist[1])])
        if sum([0 if x > 0 else 1 for x in tl]) > 0:
            if self.sys_display:
                ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d] ' + \
                     'start_pso <= end_pos'
                print(ps % (hvs, stepcount, steplen, start_line, end_line))
            return False

        # width > check_min_peak_width is considered valid mark block.
        # valid_peak_wid = tl  # tl[tl > self.check_peak_min_width]
        valid_peak_var = np.var(tl)  # np.var(tl[tl>self.check_peak_min_width])
        valid_peak_num = len(tl)
        if self.sys_run_check or self.sys_run_test:
            self.pos_peak_wid_var_log.update({(hvs[0], stepcount): valid_peak_var})

        if self.sys_run_check:
            if valid_peak_var > self.check_mark_peak_top_var:
                if self.sys_display:
                    ps = 'detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                         'num=%3d, peak_var(%4.2f) is too big'
                    print(ps % (hvs, stepcount, steplen, start_line, end_line, 0, valid_peak_var))
                return False

        # max width is too bigger than min width
        if len(tl) > 0:
            maxwid = max(tl)  # max(valid_peak_wid)
            minwid = min(tl)  # min(valid_peak_wid)
            # widratio = minwid/maxwid
            if maxwid > minwid * self.check_peak_min_max_width_ratio:
                if self.sys_display:
                    print('detect mark %s, step=%3d, steplen=%3d, zone=[%4d--%4d], num=%3d,' %
                          (hvs, stepcount, steplen, start_line, end_line, valid_peak_num),
                          ' invalid peak maxwid/minwid = %2d/%2d' % (maxwid, minwid)
                          )
                return False
        else:
            if self.sys_display:
                ps = 'check mark fail: %s, step=%3d, steplen=%3d, zone=[%4d--%4d], ' + \
                     'no valid width mark found!'
                print(ps % (hvs, stepcount, steplen, start_line, end_line))
            return False

        return True

    def _check_mark_sel_opt(self, sels: dict):     # choice best start_end_list
        wid_gap_var = {k: self._check_mark_sel_var(sels[k]) for k in sels}
        min_var = min(wid_gap_var.values())
        for k in wid_gap_var:
            if wid_gap_var[k] == min_var:
                return k
        return None

    def _check_mark_sel_opt2(self, mark_is_horizon):
        if mark_is_horizon:
            stdlog = self.pos_valid_hmapfun_std_log
        else:
            stdlog = self.pos_valid_vmapfun_std_log
        opt_t = [stdlog[st][0]*0.6 + 100*50/stdlog[st][1]*0.4 for st in stdlog]
        return list(stdlog.keys())[np.where(opt_t == np.max(opt_t))[0][0]]

    @staticmethod
    def _check_mark_sel_var(sel: list):  # start_end_list
        # sel = rc.model.pos_start_end_list_log[k]
        result = 10000
        if (len(sel[0]) == len(sel[1])) & (len(sel[0]) > 2):
            wids = [y - x for x, y in zip(sel[0], sel[1])]
            gaps = [x - y for x, y in zip(sel[0][1:], sel[1][0:-1])]
            if len(wids) > 0:
                # result = 0.6 * stt.describe(wids).variance + 0.4 * stt.describe(gap).variance
                return 0.6 * np.var(wids) + 0.4 * np.var(gaps)
        return result

    def _check_mark_pos_byconv(self, pixel_map_vec) -> tuple:

        # cluster mapfun points to peak points and valley points
        svec = [[x] for x in pixel_map_vec]
        cl = KMeans(2)
        cl.fit(svec)

        # deplicated method: use cl.predict(svec)
        '''
        # use .predict to cause some peak recog err because of low peak points clustered into valley(0)
        # when some sharp points appear. for example f22.filelist[42] horizon mark recog
        if 1 == 2:
            pixel_map_vec01 = cl.predict(svec)
            if cl.cluster_centers_[0] > cl.cluster_centers_[1]:
                pixel_map_vec01 = [0 if x > 0 else 1 for x in pixel_map_vec01]
        '''

        # use mean * gold_seg from cluster_centers of kmeans model
        pixel_map_vec01 = pixel_map_vec
        gold_seg = 0.618  # not 0.618
        img_zone_pixel_map_mean = cl.cluster_centers_.mean()
        pixel_map_vec01[pixel_map_vec < img_zone_pixel_map_mean * gold_seg] = 0
        pixel_map_vec01[pixel_map_vec >= img_zone_pixel_map_mean * gold_seg] = 1

        # smooth sharp peak and valley.
        pixel_map_vec01 = self._check_mark_mapfun_smoothsharp(pixel_map_vec01)

        # check mark positions. with opposite direction in convolve template
        mark_start_template = np.array([1, 1, 1, -1])
        mark_end_template = np.array([-1, 1, 1, 1])
        judg_value = 3
        r1 = np.convolve(pixel_map_vec01, mark_start_template, 'valid')
        r2 = np.convolve(pixel_map_vec01, mark_end_template, 'valid')

        # mark_position = np.where(r == 3), center point is the pos
        return [np.where(r1 == judg_value)[0] + 1, np.where(r2 == judg_value)[0] + 2], pixel_map_vec01

    # not use now
    '''
    def cluster_kmeans(self, cluster_num, sample_list):
        self.omr_kmeans_cluster.n_clusters = cluster_num
        self.omr_kmeans_cluster.fit(sample_list)
        rs = self.omr_kmeans_cluster.predict(sample_list)
        if self.omr_kmeans_cluster.cluster_centers_[0][0] > self.omr_kmeans_cluster.cluster_centers_[1][0]:
            #rs = [0 if x > 0 else 1 for x in rs]
            rs = 1 - rs
        return rs
    '''

    # decapitated
    def _check_mark_peak_adjust(self):
        # not neccessary
        # return
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('adjust peak fail: no position vector created!')
            return
        peaknum = len(self.pos_xy_start_end_list[0])
        pw = np.array([self.pos_xy_start_end_list[1][i] - self.pos_xy_start_end_list[0][i]
                       for i in range(peaknum)])
        vw = np.array([self.pos_xy_start_end_list[0][i + 1] - self.pos_xy_start_end_list[1][i]
                       for i in range(peaknum - 1)])
        mpw = int(pw.mean())
        mvw = int(vw.mean())
        # reduce wider peak
        for i in range(peaknum - 1):
            if pw[i] > mpw + 3:
                if vw[i] < mvw:
                    self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] - (pw[i] - mpw)
                    self.pos_x_prj_list[self.pos_xy_start_end_list[1][i]:self.pos_xy_start_end_list[1][i] + mpw] = 0
                else:
                    self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] + (pw[i] - mpw)
                    self.pos_x_prj_list[self.pos_xy_start_end_list[0][i] - mpw:self.pos_xy_start_end_list[1][i]] = 0
        # move peak
        vw = [self.pos_xy_start_end_list[0][i + 1] - self.pos_xy_start_end_list[1][i] for i in range(peaknum - 1)]
        # not move first and last peak
        for i in range(1, peaknum-1):
            # move left
            if vw[i-1] > vw[i] + 3:
                self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] - 3
                self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] - 3
                self.pos_x_prj_list[self.pos_xy_start_end_list[0][i]:self.pos_xy_start_end_list[0][i] + 3] = 1
                self.pos_x_prj_list[self.pos_xy_start_end_list[1][i]:self.pos_xy_start_end_list[1][i] + 3 + 1] = 0
                if self.sys_display:
                    print(f'move peak{i} to left')
            # move right
            if vw[i] > vw[i-1] + 3:
                self.pos_xy_start_end_list[0][i] = self.pos_xy_start_end_list[0][i] + 3
                self.pos_xy_start_end_list[1][i] = self.pos_xy_start_end_list[1][i] + 3
                self.pos_x_prj_list[self.pos_xy_start_end_list[0][i] - 3:self.pos_xy_start_end_list[0][i]] = 0
                self.pos_x_prj_list[self.pos_xy_start_end_list[1][i] - 3:self.pos_xy_start_end_list[1][i]] = 1
                if self.sys_display:
                    print(f'move peak{i} to right')

    def _check_mark_mapfun_smoothsharp(self, mapfun01):

        _mapfun01 = np.copy(mapfun01)

        # remove sharp peak with -1, 1*j, -1
        # stop = 0
        while True:
            stop = 0
            for j in range(1, self.check_peak_min_width+1):
                smooth_template = [-1] + [1] * j + [-1]
                ck = np.convolve(_mapfun01, smooth_template, 'valid')
                find_pos = np.where(ck == j)[0]
                if len(find_pos) > 0:
                    _mapfun01[find_pos[0]+1:find_pos[0]+1+j+1] = 0
                else:
                    stop = stop + 1
            if stop == self.check_peak_min_width:
                break

        # fill sharp valley 101, 1001
        for j in range(1, 3):
            smooth_template = [1] + [-1] * j + [1]
            ck = np.convolve(_mapfun01, smooth_template, 'valid')
            find_pos = np.where(ck == 2)[0]
            if len(find_pos) > 0:
                _mapfun01[find_pos[0]+1:find_pos[0]+1+j] = 1

        # remove start down and end up semi-peak
        for j in range(10, 1, -1):
            if sum(_mapfun01[:j]) == j:
                _mapfun01[:j] = 0
                break
        for j in range(-1, -11, -1):
            if sum(_mapfun01[j:]) == -j:
                _mapfun01[j:] = 0
                break

        return _mapfun01

    def _check_mark_tilt(self):
        if not self.omr_form_do_tilt_check:
            if self.sys_display:
                print('mark pos not be set in card_form[mark_format] for tilt check!')
            return
        x = [len(y) for y in self.pos_xy_start_end_list]
        if min(x) == 0:
            if self.sys_display:
                print('*: position check error!')
            return

        # horizon tilt check only need vertical move to adjust
        row = self.omr_form_mark_location_row_no - 1
        for blocknum in range(self.omr_form_mark_area['mark_horizon_number']):
            mean_list = []
            for m in range(-10, 10):
                mt = self._get_block_image_by_move((row, blocknum), 0, m)
                if len(mt) > 0:
                    mean_list.append(mt.mean())
                else:
                    mean_list.append(0)
                    if self.sys_display:
                        print('block moving err: file=%s, row=%d, num=%d' %
                              (self.image_filename, row, blocknum))
            max_mean = int(max(mean_list))
            if max_mean > mean_list[10]:  # need adjust
                move_step = np.where(np.array(mean_list) >= max_mean)[0][0]
                self.omr_result_horizon_tilt_rate[blocknum] = move_step - 10

        # vertical tilt check only need horizonal move to adjust
        col = self.omr_form_mark_location_col_no - 1
        for blocknum in range(self.omr_form_mark_area['mark_vertical_number']):
            mean_list = []
            for m in range(-10, 10):
                mt = self._get_block_image_by_move((blocknum, col), m, 0)
                if min(mt.shape) > 0:
                    mean_list.append(mt.mean())
                else:
                    mean_list.append(0)
                    if self.sys_display:
                        print('block moving err: file=%s, col=%d, num=%d' %
                              (self.image_filename, col, blocknum))
            # if len(mean_list) > 0:
            max_mean = max(mean_list)
            if max_mean > mean_list[10]:  # need adjust
                move_step = np.where(np.array(mean_list) >= max_mean)[0][0]
                self.omr_result_vertical_tilt_rate[blocknum] = move_step - 10

    def _get_block_image_by_move(self, block_coord_row_col, block_move_horizon, block_move_vertical):
        # print(self.pos_xy_start_end_list, block_coord_row_col)
        block_left = self.pos_xy_start_end_list[0][block_coord_row_col[1]]
        block_top = self.pos_xy_start_end_list[2][block_coord_row_col[0]]
        block_width = self.pos_xy_start_end_list[1][block_coord_row_col[1]] - \
            self.pos_xy_start_end_list[0][block_coord_row_col[1]]
        block_high = self.pos_xy_start_end_list[3][block_coord_row_col[0]] - \
            self.pos_xy_start_end_list[2][block_coord_row_col[0]]
        # block_high, block_width = self.omr_result_coord_markimage_dict[block_coord_row_col].shape
        return self.image_card_2dmatrix[block_top+block_move_vertical:block_top+block_high+block_move_vertical,
                                        block_left+block_move_horizon:block_left+block_width+block_move_horizon]

    def _get_coord_blockimage_dict(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('create omrdict fail:no position vector created!')
            return

        # valid area: cut area for painting points
        for x in range(self.omr_form_valid_area['mark_horizon_number'][0]-1,
                       self.omr_form_valid_area['mark_horizon_number'][1]):
            for y in range(self.omr_form_valid_area['mark_vertical_number'][0]-1,
                           self.omr_form_valid_area['mark_vertical_number'][1]):
                if (y, x) in self.omr_form_coord_group_dict:
                    x_tilt = self.omr_result_horizon_tilt_rate[x]
                    y_tilt = self.omr_result_vertical_tilt_rate[y]
                    self.omr_result_coord_blockimage_dict[(y, x)] = \
                        self.image_card_2dmatrix[
                            self.pos_xy_start_end_list[2][y] + x_tilt - self.check_block_y_extend:
                            self.pos_xy_start_end_list[3][y] + x_tilt + 1 + self.check_block_y_extend,
                            self.pos_xy_start_end_list[0][x] + y_tilt - self.check_block_x_extend:
                            self.pos_xy_start_end_list[1][x] + y_tilt + 1 + self.check_block_x_extend]
        '''
        # mark area: mark edge points
        for x in range(self.omr_form_mark_area['mark_horizon_number']):
            for y in range(self.omr_form_mark_area['mark_vertical_number']):
                x_tilt = self.omr_result_horizon_tilt_rate[x]
                y_tilt = self.omr_result_vertical_tilt_rate[y]
                self.omr_result_coord_markimage_dict[(y, x)] = \
                    self.image_card_2dmatrix[self.pos_xy_start_end_list[2][y] + x_tilt:
                                             self.pos_xy_start_end_list[3][y] + 1 + x_tilt,
                                             self.pos_xy_start_end_list[0][x] + y_tilt:
                                             self.pos_xy_start_end_list[1][x] + 1 + y_tilt]
       '''

    def _get_block_features_with_moving(self, bmat, row, col):

        # depcated now, if using tilt check
        if not self.check_block_by_floating:
            return self._get_block_features(bmat)

        # float steplen=2, not optimizing method
        xs = self.pos_xy_start_end_list[2][row]
        xe = self.pos_xy_start_end_list[3][row] + 1
        ys = self.pos_xy_start_end_list[0][col]
        ye = self.pos_xy_start_end_list[1][col] + 1
        # origin
        sa = self._get_block_features(bmat)
        if sa[0] > 120:
            return sa
        # move left
        bmat = self.image_card_2dmatrix[xs:xe, ys - 2:ye - 2]
        sa2 = self._get_block_features(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move right
        bmat = self.image_card_2dmatrix[xs:xe, ys + 2:ye + 2]
        sa2 = self._get_block_features(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move up
        bmat = self.image_card_2dmatrix[xs - 2:xe - 2, ys:ye]
        sa2 = self._get_block_features(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        # move down
        bmat = self.image_card_2dmatrix[xs + 2:xe + 2, ys:ye]
        sa2 = self._get_block_features(bmat)
        if sa2[0] > sa[0]:
            sa = sa2
        return sa

    def _get_block_features(self, blockmat):

        if len(blockmat) == 0:
            return 0, 0, 0, 0

        th = self.check_gray_threshold

        # feature1: mean level
        # use coefficient 10/255 as weight-coeff
        # coeff0 = 10/255 = 2/51 --> 1/25
        feat01 = round(blockmat.mean() / 25, 3)

        # feature2: big-mean-line_ratio in row or col
        # use omr_threshold to judge painting area saturation
        # row mean and col mean compare
        rowmean = blockmat.mean(axis=0)
        colmean = blockmat.mean(axis=1)
        feat02 = round(len(rowmean[rowmean > th]) / len(rowmean), 3)
        feat03 = round(len(colmean[colmean > th]) / len(colmean), 3)

        # feature3: big-pixel-ratio
        bignum = len(blockmat[blockmat > th])
        feat04 = round(bignum / blockmat.size, 3)

        # feature4: hole-number
        # st05 = self.fun_detect_hole(block01)
        # feat05 = 0

        # saturational area is more than 3
        # th = self.check_gray_threshold  # 50

        # feature5: saturation area exists
        # st06 = cv2.filter2D(p, -1, np.ones([3, 5]))
        # feat06 = filters.convolve(self.fun_normto01(blockmat, th),
        #                        np.ones([3, 5]), mode='constant')
        # feat06 = 1 if len(st06[st06 >= 14]) >= 1 else 0

        return feat01, feat02, feat03, feat04  # , feat05, feat06

    @staticmethod
    def fun_detect_hole(mat):
        # 3x4 hole
        m = np.array([[1,  1,  1,  1],
                      [1, -1, -1,  1],
                      [1,  1,  1,  1]])
        rf = filters.convolve(mat, m, mode='constant')
        r0 = len(rf[rf == 10])
        # r0 = 1 if len(rf[rf == 10]) > 0 else 0
        # 3x5 hole
        m = np.array([[1,  1,  1,  1, 1],
                     [1, -1, -1, -1, 1],
                     [1,  1,  1,  1, 1]])
        rf = filters.convolve(mat, m, mode='constant')
        r1 = len(rf[rf == 12])
        if r1 == 0:
            m = np.array([[1,  1,  1, 1, 1],
                         [1, -1, -1, 0, 1],
                         [1,  1,  1, 1, 1]])
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        if r1 == 0:
            m = np.array([[1,  1,  1,  1, 1],
                         [1,  0, -1, -1, 1],
                         [1,  1,  1,  1, 1]])
            rf = filters.convolve(mat, m, mode='constant')
            r1 = len(rf[rf == 12])
        # 4x5 hole
        m = np.array([[0,  1,  1,  1, 0],
                     [1,  0, -1, -1, 1],
                     [1,  0, -1, -1, 1],
                     [0,  1,  1,  1, 0]])
        rf = filters.convolve(mat, m, mode='constant')
        r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1, 1, 0],
                         [1, -1, -1, 0, 1],
                         [1, -1, -1, 0, 1],
                         [0,  1,  1, 1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1, 1, 0],
                         [1,  0,  0, 0, 1],
                         [1, -1, -1, -1, 1],
                         [0,  1,  1, 1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        if r2 == 0:
            m = np.array([[0,  1,  1,  1, 0],
                         [1, -1, -1, -1, 1],
                         [1,  0,  0,  0, 1],
                         [0,  1,  1,  1, 0]])
            rf = filters.convolve(mat, m, mode='constant')
            r2 = len(rf[rf == 10])
        return r0 + (1 if r1 > 0 else 0) + (1 if r2 > 0 else 0)

    def _get_image_with_rawblocks(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('no position vector created! so cannot create omrdict!')
            return
        # create a blackgroud model for omr display image
        omrimage = np.zeros(self.image_card_2dmatrix.shape)
        for col in range(self.omr_form_mark_area['mark_horizon_number']):
            for row in range(self.omr_form_mark_area['mark_vertical_number']):
                omrimage[self.pos_xy_start_end_list[2][row]: self.pos_xy_start_end_list[3][row] + 1,
                         self.pos_xy_start_end_list[0][col]: self.pos_xy_start_end_list[1][col] + 1] = \
                    self.image_card_2dmatrix[self.pos_xy_start_end_list[2][row]:
                                             self.pos_xy_start_end_list[3][row] + 1,
                                             self.pos_xy_start_end_list[0][col]:
                                             self.pos_xy_start_end_list[1][col] + 1]
        # self.image_recog_blocks = omrimage
        self.image_blackground_with_rawblock = omrimage
        return omrimage

    def _get_image_with_recogblocks(self):
        lencheck = len(self.pos_xy_start_end_list[0]) * len(self.pos_xy_start_end_list[1]) * \
                   len(self.pos_xy_start_end_list[3]) * len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('no position vector created! so cannot create recog_omr_image!')
            return
        # init image with zero background
        omr_recog_block = np.zeros(self.image_card_2dmatrix.shape)
        # set block_area with painted block in raw image
        for label, coord in zip(self.omr_result_data_dict['label'], self.omr_result_data_dict['coord']):
            if label == 1:
                if coord in self.omr_form_coord_group_dict:
                    omr_recog_block[self.pos_xy_start_end_list[2][coord[0]] - self.check_block_y_extend:
                                    self.pos_xy_start_end_list[3][coord[0]] + 1 + self.check_block_y_extend,
                                    self.pos_xy_start_end_list[0][coord[1]] - self.check_block_x_extend:
                                    self.pos_xy_start_end_list[1][coord[1]] + 1 + self.check_block_x_extend] \
                        = self.omr_result_coord_blockimage_dict[coord]
        self.image_blackground_with_recogblock = omr_recog_block
        return omr_recog_block

    # create recog_data, and test use svm in sklearn
    def _get_result_data_dict(self):
        # init data dict
        self.omr_result_data_dict = \
            {'coord': [], 'feature': [], 'group': [],
             'code': [], 'mode': [], 'label': []}

        # checked in poslist evaluate
        '''
        # check pos_list
        lencheck = \
            len(self.pos_xy_start_end_list[0]) * \
            len(self.pos_xy_start_end_list[1]) * \
            len(self.pos_xy_start_end_list[3]) * \
            len(self.pos_xy_start_end_list[2])
        invalid_result = (lencheck == 0) | \
                         (len(self.pos_xy_start_end_list[0]) != len(self.pos_xy_start_end_list[1])) | \
                         (len(self.pos_xy_start_end_list[2]) != len(self.pos_xy_start_end_list[3]))
        if invalid_result:
            if self.sys_display:
                print('create recog_data fail: no position vector!')
            return
        '''

        # create result_data_dict
        for j in range(self.omr_form_valid_area['mark_horizon_number'][0]-1,
                       self.omr_form_valid_area['mark_horizon_number'][1]):
            for i in range(self.omr_form_valid_area['mark_vertical_number'][0]-1,
                           self.omr_form_valid_area['mark_vertical_number'][1]):
                if (i, j) in self.omr_form_coord_group_dict:
                    self.omr_result_data_dict['coord'].append((i, j))
                    self.omr_result_data_dict['feature'].append(
                        self._get_block_features(self.omr_result_coord_blockimage_dict[(i, j)]))
                    self.omr_result_data_dict['group'].append(self.omr_form_coord_group_dict[(i, j)][0])
                    self.omr_result_data_dict['code'].append(self.omr_form_coord_group_dict[(i, j)][1])
                    self.omr_result_data_dict['mode'].append(self.omr_form_coord_group_dict[(i, j)][2])

        # cluster method to classify block to painting or not
        cluster_method = 2
        label_result = []

        # cluster.kmeans trained in group
        # result: no cards with loss recog, 4 cards with multi_recog(over)
        if cluster_method == 1:
            gpos = 0
            for g in self.omr_form_group_dict:
                self.omr_kmeans_cluster.fit(self.omr_result_data_dict['feature'])
                glen = self.omr_form_group_dict[g][1]
                label_result += \
                    list(self._cluster_block(self.omr_result_data_dict['feature'][gpos: gpos+glen]))
                gpos = gpos + glen
            # self.omr_result_data_dict['label'] = label_result

        # cluster.kmeans trained in card,
        # testf21: 2 cards with loss recog, 1 card with multi_recog
        # testf22: 9 cards with loss recog, 4 cards with multi recog(19, 28, 160, 205)
        # effection is the best now(2018-2-27)
        if cluster_method == 2:
            self.omr_kmeans_cluster.n_clusters = 2
            self.omr_kmeans_cluster.fit(self.omr_result_data_dict['feature'])
            label_result = self._cluster_block(self.omr_result_data_dict['feature'])

        # cluster.kmeans in card_set(223) training model: 19 cards with loss_recog, no cards with multi_recog(over)
        if cluster_method == 3:
            self.omr_kmeans_cluster = jb.load('model_kmeans_im21.m')
            label_result = self._cluster_block(self.omr_result_data_dict['feature'])

        # cluster.kmeans by card_set(223)(42370groups): 26 cards with loss_recog, no cards with multi_recog(over)
        if cluster_method == 4:
            self.omr_kmeans_cluster = jb.load('model_kmeans_im22.m')
            label_result = self._cluster_block(self.omr_result_data_dict['feature'])

        # cluster.svm trained by cardset223(41990groups), result: 19 cards with loss recog, no cards with multirecog
        if cluster_method == 5:
            self.omr_kmeans_cluster = jb.load('model_svm_im21.m')
            label_result = self.omr_kmeans_cluster.predict(self.omr_result_data_dict['feature'])

        # cluster use cnn model m18test trained by omrimages set 123, loss too much in y18-f109
        # if cluster_method == 6:
        #    group_coord_image_list = [self.omr_result_coord_blockimage_dict[coord]
        #                              for coord in self.omr_result_data_dict['coord']]
        #    label_result = self.cnnmodel.predict_rawimage(group_coord_image_list)

        self.omr_result_data_dict['label'] = label_result

    def _cluster_block(self, feats):
        # cl.fit(feats)
        label_result = self.omr_kmeans_cluster.predict(feats)
        centers = self.omr_kmeans_cluster.cluster_centers_
        if centers[0, 0] > centers[1, 0]:   # gray mean level low for 1
            label_result = 1 - label_result
        for fi, fe in enumerate(feats):
            if fe[0] < self.check_block_min_gray_mean/25:    # gray_level = 0.5*25 = 12.5
                label_result[fi] = 0
        return label_result

    # result dataframe
    def _set_result_dataframe_default(self):

        # singular result: '***'=error, '...'=blank
        # result_info: record error choice in mode('M', 'S'), gno:[result_str for group]
        # score_group format: group_no=score,...

        # default dataframe
        if 'score_format' in self.form:
            if self.form['score_format']['do_score']:
                self.omr_result_dataframe = \
                    pd.DataFrame({'card': [Util.find_file_from_pathfile(self.image_filename).split('.')[0]],
                                  'len': [-1],
                                  'result': ['*'*len(self.omr_form_group_dict)],
                                  'result_info': [''],
                                  'score': [0],
                                  'score_group': [''],
                                  }, index=[self.card_index_no])
                return
        self.omr_result_dataframe = \
            pd.DataFrame({'card': [Util.find_file_from_pathfile(self.image_filename).split('.')[0]],
                          'len': [-1],
                          'result': ['*'*len(self.omr_form_group_dict)],
                          'result_info': ['']
                          }, index=[self.card_index_no])

        self.omr_result_dataframe_groupinfo = \
            pd.DataFrame({'coord': [(-1)],
                          'label': [-1],
                          'feat': [(-1, -1, -1, -1)],
                          'group': [''],
                          'code':  [''],
                          'mode': ['']
                          })

    # result dataframe
    def _get_result_dataframe(self):

        # singular result: '***'=error, 'P'=blank
        # result_info: record error choice in mode('M', 'S'), gno:[result_str for group]
        # score_group format: group_no=score,...

        # init result dataframes
        self._set_result_dataframe_default()

        # no recog_data, return len=-1, code='***'
        if len(self.omr_result_data_dict['label']) == 0:
            if self.sys_display:
                print('result fail: recog data is not created!')
            # return self.omr_result_dataframe with -1, '***'
            return

        # create result dataframe
        rdf = pd.DataFrame({'coord': self.omr_result_data_dict['coord'],
                            'label': self.omr_result_data_dict['label'],
                            'feat': self.omr_result_data_dict['feature'],
                            'group': self.omr_result_data_dict['group'],
                            'code': self.omr_result_data_dict['code'],
                            'mode': self.omr_result_data_dict['mode']
                            })

        # feature0_var too small means to no painting, return len=0, result='...'
        feature0_var = np.var([x[0] for x in self.omr_result_data_dict['feature']])
        if feature0_var < 0.1:
            if self.sys_display:
                print('invalid features: too small blocks gray var={}'.format(feature0_var))
            self.omr_result_dataframe_groupinfo = rdf
            self.omr_result_dataframe.loc[:, 'len'] = 0
            self.omr_result_dataframe.loc[:, 'result'] = 'P' * len(self.omr_form_group_dict)
            self.omr_result_dataframe.loc[:, 'result_info'] = ''
            if 'score_format' in self.form:
                if self.form['score_format']['do_score']:
                    self.omr_result_dataframe.loc[:, 'score'] = 0
                    self.omr_result_dataframe.loc[:, 'score_group'] = ''
            return

        # set label 0 (no painted) block's code to ''
        rdf.loc[rdf.label == 0, 'code'] = ''

        # create result dataframe
        outdf = rdf[rdf.group > 0].sort_values('group')[['group', 'code']].groupby('group').sum()

        rs_codelen = 0
        rs_code = []
        group_str = ''
        if len(outdf) > 0:
            out_se = outdf['code'].apply(lambda s: ''.join(sorted(list(s))))
            group_list = sorted(self.omr_form_group_dict.keys())
            for group_no in group_list:
                if group_no in out_se.index:
                    rs = out_se[group_no]
                    if len(rs) > 0:
                        rs_codelen = rs_codelen + 1
                    # mode == 'D', 012..9 digit mode
                    if self.omr_form_group_dict[group_no][4] == 'D':
                        if len(rs) == 1:
                            ts = rs
                        elif len(rs) == 0:
                            ts = '.'
                        else:
                            ts = '>'
                        rs_code.append(ts)
                        continue
                    # mode = 'B', 8421 bcd code mode
                    if self.omr_form_group_dict[group_no][4] == 'B':
                        if rs in self.omr_encode_dict:
                            ts = self.omr_encode_dict[rs]
                        else:
                            ts = '>'
                        rs_code.append(ts)
                        continue
                    # mode = 'X'
                    if self.omr_form_group_dict[group_no][4] in ['X', 'S']:
                        if len(rs) == 0:
                            ts = 'P'
                        elif len(rs) == 1:
                            ts = rs
                        else:
                            ts = '>'
                            group_str = group_str + str(group_no) + ':' + rs + ','
                        rs_code.append(ts)
                        continue
                    # mode = 'M'
                    if rs in self.omr_encode_dict:
                        ts = self.omr_encode_dict[rs]
                    else:
                        ts = '>'
                        if (self.omr_form_group_dict[group_no][4] != 'M') and len(rs) > 1:
                            group_str = group_str + str(group_no) + ':' + rs + ','
                    rs_code.append(ts)
                else:
                    # group g not found
                    rs_code.append('?')
                    group_str = group_str + str(group_no) + ':?,'
            rs_code = ''.join(rs_code)
            group_str = group_str[:-1]
        else:
            # no group found, valid area maybe not cover group blocks!
            # return self.omr_result_dataframe with len=-1, result='***'
            return

        # debug result to debug_dataframe: fname, coord, group, label, feature
        if self.sys_run_check or self.sys_run_test:
            self.omr_result_dataframe_groupinfo = rdf

        # group result to dataframe: fname, len, group_str, result
        if 'score_format' in self.form:
            if self.form['score_format']['do_score']:
                self.omr_result_dataframe = \
                    pd.DataFrame({'card': [Util.find_file_from_pathfile(self.image_filename).split('.')[0]],
                                  'len': [rs_codelen],
                                  'result': [rs_code],
                                  'result_info': [group_str],
                                  'score_group': [''],
                                  'score': [0]
                                  }, index=[self.card_index_no])
                if len(rs_code) > 0:
                    rss = self._get_score_from_result(rs_code)
                    self.omr_result_dataframe.loc[:, 'score_group'] = rss[0]
                    self.omr_result_dataframe.loc[:, 'score'] = rss[1]
                return
        # else: no score_format or do_score=False
        self.omr_result_dataframe = \
            pd.DataFrame({'card': [Util.find_file_from_pathfile(self.image_filename).split('.')[0]],
                          'len': [rs_codelen],
                          'result': [rs_code],
                          'result_info': [group_str]
                          }, index=[self.card_index_no])

    def _get_score_from_result(self, rs):
        if len(rs) != len(self.omr_form_group_dict):
            return '***'
        ss = ''
        sep = ''
        sep1 = '='
        gs = sorted(list(self.omr_form_group_dict.keys()))
        for rc, rg in zip(rs, gs):
            if (sep == '') & (ss != ''):
                sep = ','
            if rg in self.form['score_format']['score_dict']:
                score_d = self.form['score_format']['score_dict'][rg]
                if rc in score_d:
                    ss = ss + sep + str(rg) + sep1 + str(score_d[rc])
                else:
                    ss = ss + sep + str(rg) + sep1 + '0'
            # else:
                # ss = ss + sep + str(rg) + '=*'
                # ss = ss + sep + '-1'
        ssum = 0
        if len(ss) > 0:
            ssum = sum([eval(x.split(sep1)[1]) for x in ss.split(sep) if eval(x.split(sep1)[1]) >= 0])

        return ss, ssum

    # --- show omrimage or plot result data ---
    def plot_result(self):
        plt.figure('Omr Model:'+self.image_filename)
        plt.subplot(231)
        self.plot_image_raw_card()
        plt.subplot(232)
        self.plot_image_clip_card()
        plt.subplot(233)
        self.plot_image_recogblocks()
        plt.subplot(223)
        self.plot_mapfun_horizon_mark()
        plt.subplot(224)
        self.plot_mapfun_vertical_mark()

    def plot_image_raw_card(self):
        if type(self.image_rawcard) != np.ndarray:
            if self.sys_display:
                print('no raw card image file')
            return
        plt.imshow(self.image_rawcard)

    def plot_image_clip_card(self):
        plt.imshow(self.image_card_2dmatrix)

    def plot_image_rawblocks(self):
        plt.title('recognized - omr - region ' + self.image_filename)
        plt.imshow(self._get_image_with_rawblocks())

    def plot_image_recogblocks(self):
        if type(self.image_blackground_with_recogblock) != np.ndarray:
            self._get_image_with_recogblocks()
        plt.imshow(self.image_blackground_with_recogblock)

    def plot_image_with_markline(self):
        plt.imshow(self.image_card_2dmatrix)
        xset = np.concatenate([self.pos_xy_start_end_list[0], self.pos_xy_start_end_list[1]])
        yset = np.concatenate([self.pos_xy_start_end_list[2], self.pos_xy_start_end_list[3]])
        xrange = [x for x in range(self.image_card_2dmatrix.shape[1])]
        yrange = [y for y in range(self.image_card_2dmatrix.shape[0])]
        for x in xset:
            plt.plot([x]*len(yrange), yrange)
        for y in yset:
            plt.plot(xrange, [y]*len(xrange))
        for p, xl in enumerate(self.pos_xy_start_end_list[0]):
            plt.text(xl, -6, '%2d' % (p+1))
            plt.text(xl, self.image_card_2dmatrix.shape[0]+10, '%2d' % (p+1))
        for p, yl in enumerate(self.pos_xy_start_end_list[2]):
            plt.text(-6, yl, '%2d' % (p+1))
            plt.text(self.image_card_2dmatrix.shape[1]+2, yl, '%2d' % (p+1))

    def plot_mapfun_horizon_mark(self):
        plt.xlabel('horizon mark map fun')
        plt.plot(self.pos_x_prj_list)

    def plot_mapfun_vertical_mark(self):
        plt.xlabel('vertical mark map fun')
        plt.plot(self.pos_y_prj_list)

    def plot_grid_with_blockpoints(self):
        from pylab import subplot, scatter, gca, show
        from matplotlib.ticker import MultipleLocator  # , FormatStrFormatter
        # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        plt.figure('markgrid')
        plt.title(self.image_filename)
        data_coord = np.array(self.omr_result_data_dict['coord']) + 1
        x, y, z = [], [], []
        for i, lab in enumerate(self.omr_result_data_dict['label']):
            if lab == 1:
                x.append(data_coord[i, 0])
                y.append(data_coord[i, 1])
                # z.append(data_mean[i])
        xy_major_locator = MultipleLocator(5)  # 主刻度标签设置为5的倍数
        xy_minor_locator = MultipleLocator(1)  # 副刻度标签设置为1的倍数

        ax = subplot(111)
        ax.xaxis.set_major_locator(xy_major_locator)
        ax.xaxis.set_minor_locator(xy_minor_locator)
        ax.yaxis.set_major_locator(xy_major_locator)
        ax.yaxis.set_minor_locator(xy_minor_locator)
        ax.xaxis.set_ticks(np.arange(1, self.omr_form_mark_area['mark_horizon_number'], 1))
        ax.yaxis.set_ticks(np.arange(1, self.omr_form_mark_area['mark_vertical_number'], 5))

        scatter(y, x)  # , c=z, cmap=cm)
        gca().invert_yaxis()

        # gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1d'))
        ax.xaxis.grid(b=True, which='minor')  # , color='red', linestyle='dashed')      # x坐标轴的网格使用副刻度
        ax.yaxis.grid(b=True, which='minor')  # , color='green', linestyle='dashed')    # y坐标轴的网格使用主刻度
        ax.grid(color='gray', linestyle='-', linewidth=2)
        show()


# --- some useful functions in omrmodel or outside
class Util:

    def __init__(self):
        pass

    @staticmethod
    def show_image(fstr):
        if os.path.isfile(fstr):
            plt.imshow(mg.imread(fstr))
            plt.title(fstr)
            plt.show()
        else:
            print('file \"%s\" is not found!' % fstr)

    @staticmethod
    def find_file_from_pathfile(path_file):
        return path_file.replace('/', '\\').split('\\')[-1]

    @staticmethod
    def find_path_from_pathfile(path_file):
        ts = Util.find_file_from_pathfile(path_file)
        return path_file.replace(ts, '').replace('\\', '/')

    @staticmethod
    def glob_files_from_path(path, substr_list):
        if type(substr_list) == str:
            if len(substr_list) == 0:
                substr_list = []
            else:
                substr_list = [substr_list]
        if not os.path.isdir(path):
            return ['path not exist!']
        file_list = []
        for f in glob.glob(path+'/*'):
            # print(f)
            if os.path.isfile(f):
                #if len(substr_list) == 0:
                #    file_list.append(f)
                if sum([1 if s in f else 0 for s in substr_list]) == len(substr_list):  # now=&, think |
                    file_list.append(f)
            if os.path.isdir(f):
                [file_list.append(s)
                 for s in Util.glob_files_from_path(path=f, substr_list=substr_list)]
        return file_list

    @staticmethod
    def matrix_row_reverse(matrix_2d):
        return matrix_2d[::-1]

    @staticmethod
    def matrix_col_reverse(matrix_2d):
        return np.array([matrix_2d[r, :][::-1] for r in range(matrix_2d.shape[0])])

    @staticmethod
    def matrix_rotate90_right(matrix_2d):
        # matrix_2d[:] = np.array(map(list, zip(*matrix_2d[::-1])))
        temp = map(list, zip(*matrix_2d[::-1]))
        return np.array(list(temp))

    @staticmethod
    def find_high_count_element(mylist: list):
        cn = Counter(mylist)
        if len(cn) > 0:
            return cn.most_common(1)[0][0]
        else:
            return 0

    @staticmethod
    def find_high_count_continue_element(mylist: list):
        if len(mylist) == 0:
            print('empty list')
            return -1
        countlist = [0 for _ in mylist]
        for i, e in enumerate(mylist):
            for ee in mylist[i:]:
                if ee == e:
                    countlist[i] += 1
                else:
                    break
        m = max(countlist)
        p = countlist.index(m)
        return mylist[p]

    @staticmethod
    def omr_dataframe_group_to_dict(g):
        g = g.split(sep=',')
        return {eval(v.split(':')[0]): v.split(':')[1] for v in g}

    @staticmethod
    def cluster_block(cl, feats):
        # cl.fit(feats)
        label_result = cl.predict(feats)
        centers = cl.cluster_centers_
        if centers[0, 0] > centers[1, 0]:   # gray mean level low for 1
            # label_result = [0 if x > 0 else 1 for x in label_result]
            label_result = 1 - label_result

        for fi, fe in enumerate(feats):
            if fe[0] < 0.35:
                label_result[fi] = 0

        return label_result

    @staticmethod
    def softmax(vector):
        sumvalue = sum([np.exp(v) for v in vector])
        return [np.exp(v)/sumvalue for v in vector]

    @staticmethod
    def seek_valley_wid_from_mapfun(mf):
        mfm = np.mean(mf)
        r = []
        va = 0
        for x in mf:
            if x < mfm:
                va = va + 1
            elif va > 0:
                r.append(va)
                va = 0
        return r

    @staticmethod
    def show_dataframe(df):
        pp.pprint(df)


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s=''):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        if len(s) > 0:
            print(s)
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('>' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


class SklearnModel:

    classify_number = 2

    def __init__(self):
        self.sample_test_ratio = 0.85
        self.data_features = None
        self.data_labels = None
        self.test_features = None
        self.test_labels = None
        self.model_dict = {
            'bayes': SklearnModel.naive_bayes_classifier,
            'svm': SklearnModel.svm_classifier,
            'knn': SklearnModel.knn_classifier,
            'logistic_regression': SklearnModel.logistic_regression_classifier,
            'random_forest': SklearnModel.random_forest_classifier,
            'decision_tree': SklearnModel.decision_tree_classifier,
            'gradient_boosting': SklearnModel.gradient_boosting_classifier,
            'svm_cross': SklearnModel.svm_cross_validation,
            'kmeans': SklearnModel.kmeans_classifier,
            'mlp': SklearnModel.mlp_classifier
           }
        self.model = None
        self.test_result_labels = None
        self.model_test_result = dict({'suc_ratio': 0, 'err_num': 0})

    def set_data(self, data_feat, data_label):
        if data_label is not None:
            data_len = len(data_feat)
            train_len = int(data_len * self.sample_test_ratio)
            test_len = data_len - train_len
            self.data_features = data_feat[0:train_len]
            self.data_labels = data_label[0:train_len]
            self.test_features = data_feat[train_len:train_len + test_len]
            self.test_labels = data_label[train_len:train_len+test_len]
        else:
            self.data_features = data_feat

    def train_model(self, model_name='kmeans'):
        if model_name not in self.model_dict:
            print('error model name:{0} in {1}'.format(model_name, self.model_dict.keys()))
            return False
        if self.data_features is None:
            print('data is not ready:', model_name)
            return False
        self.model = self.model_dict[model_name](self.data_features, self.data_labels)
        if self.test_labels is not None:
            self.test_result_labels = self.model.predict(self.test_features)
            sucnum = sum([1 if x == y else 0 for x, y in zip(self.test_labels, self.test_result_labels)])
            self.model_test_result['suc_ratio'] = sucnum / len(self.test_labels)
            self.model_test_result['err_num'] = len(self.test_labels) - sucnum
        return True

    def test_model(self, testdata_feat, testdata_label):
        model_test_result = dict({'suc_ratio': 0, 'err_num': 0})
        test_result_labels = self.model.predict(testdata_feat)
        test_result = [1 if x == y else 0 for x, y in zip(testdata_label, test_result_labels)]
        sucnum = sum(test_result)
        model_test_result['suc_ratio'] = sucnum / len(testdata_feat)
        model_test_result['err_num'] = len(testdata_feat) - sucnum
        model_test_result['err_feat'] = [{'feat': testdata_feat[i],
                                          'label': testdata_label[i],
                                          'test_label': test_result_labels[i]}
                                         for i, x in enumerate(test_result) if x == 0]
        pp.pprint(model_test_result)

    def save_model(self, pathfile='model_name_xxx.m'):
        jb.dump(self.model, pathfile)

    @staticmethod
    # Multinomial Naive Bayes Classifier
    def kmeans_classifier(train_x, train_y):
        from sklearn.cluster import KMeans
        model = KMeans(SklearnModel.classify_number)
        if train_y is None:
            model.fit(train_x)
        else:
            model.fit(train_x, train_y)
        return model

    @staticmethod
    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(train_x, train_y):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # KNN Classifier
    def knn_classifier(train_x, train_y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # Logistic Regression Classifier
    def logistic_regression_classifier(train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # Random Forest Classifier
    def random_forest_classifier(train_x, train_y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=8)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # Decision Tree Classifier
    def decision_tree_classifier(train_x, train_y):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # SVM Classifier
    def svm_classifier(train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    # SVM Classifier using cross validation
    def svm_cross_validation(train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in list(best_parameters.items()):
            print(para, val)
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        model.fit(train_x, train_y)
        return model

    @staticmethod
    def mlp_classifier(train_x, train_y):
        # 多层线性回归 linear neural network
        from sklearn.neural_network import MLPRegressor
        # solver='lbfgs',  MLP的L-BFGS在小数据上表现较好，Adam较为鲁棒
        # SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
        # alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
        # hidden_layer_sizes=(5, 2) hidden层2层, 第一层5个神经元，第二层2个神经元)，2层隐藏层，也就有3层神经网络
        clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(train_x, train_y)
        return clf


'''
class OmrCnnModel:
    def __init__(self):
        self.model_path_name_office = 'f:/studies/juyunxia/omrmodel/omr_model'
        self.model_path_name_dell = 'd:/work/omrmodel/omr_model'
        self.default_model_path_name = self.model_path_name_dell
        self.model_path_name = None
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.saver = None
        self.input_x = None
        self.input_y = None
        self.keep_prob = None
        self.y = None
        self.a = None

    def __del__(self):
        self.sess.close()

    def load_model(self, _model_path_name=''):
        if len(_model_path_name) == 0:
            self.model_path_name = self.default_model_path_name
        else:
            self.model_path_name = _model_path_name
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.model_path_name + '.ckpt.meta')
            self.saver.restore(self.sess, self.model_path_name+'.ckpt')
            self.y = tf.get_collection('predict_label')[0]
            self.a = tf.get_collection('accuracy')[0]
            self.input_x = self.graph.get_operation_by_name('input_omr_images').outputs[0]
            self.input_y = self.graph.get_operation_by_name('input_omr_labels').outputs[0]
            self.keep_prob = self.graph.get_operation_by_name('keep_prob').outputs[0]
            # yp = self.sess.run(self.y, feed_dict={self.input_x: omr_image_set, self.keep_prob: 1.0})
            # return yp

    def test(self, omr_data_set):
        with self.graph.as_default():
            # 测试, 计算识别结果及识别率
            yp = self.sess.run(self.y, feed_dict={self.input_x: omr_data_set[0], self.keep_prob: 1.0})
            ac = self.sess.run(self.a, feed_dict={self.input_x: omr_data_set[0],
                                                  self.input_y: omr_data_set[1],
                                                  self.keep_prob: 1.0})
            print(f'accuracy={ac}')
            yr = [(1 if v[0] < v[1] else 0, i) for i, v in enumerate(yp)]
            # if [1 if v[0] < v[1] else 0][0] != omr_data_set[1][1]]
            err = [v1 for v1, v2 in zip(yr, omr_data_set[1]) if v1[0] != v2[1]]
        return err

    def predict(self, omr_image_set):
        with self.graph.as_default():
            # 使用 y 进行预测
            yp = self.sess.run(self.y, feed_dict={self.input_x: omr_image_set, self.keep_prob: 1.0})
        return yp

    def predict_rawimage(self, omr_image_set):
        norm_image_set = [cv2.resize(im/255, (12, 16), cv2.INTER_NEAREST).reshape(192)
                          for im in omr_image_set]
        with self.graph.as_default():
            # 使用 y 进行预测
            yp = self.sess.run(self.y, feed_dict={self.input_x: norm_image_set, self.keep_prob: 1.0})
        plabel = [0 if x[0] > x[1] else 1 for x in yp]
        return plabel
'''
