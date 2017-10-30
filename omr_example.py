# *_* utf-8 *_*

import omr_lib1 as ol
import glob
# import numpy as np


loc = '3-2'  # 'surface'
card1_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg' \
    if loc == 'surface' else \
    'F:\\studies\\juyunxia\\omrimage1\\*.jpg'
card2_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\*.jpg' \
    if loc == 'surface' else \
    'F:\\studies\\juyunxia\\omrimage2\\*.jpg'
filter101 = ['1-']
filter102 = ['2-']
card10x_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr0\\*.jpg' \
    if loc == 'surface' else \
    'F:\\studies\\juyunxia\\omrimage2\\*.jpg'


def omr_read(card_no):
    if card_no == 1:
        card_form = omr_form1()
    elif card_no == 2:
        card_form = omr_form2()
    elif card_no == 3:
        card_form = omr_form3()
    elif card_no == 101:
        card_form = omr_form101()
    elif card_no == 102:
        card_form = omr_form102()
    else:
        print('no this card:{0}'.format(card_no))
        return
    return ol.omr_read_batch(card_form)


def omr_form1():
    omr_image_location = card1_location
    omr_image_list = glob.glob(omr_image_location)
    group1 = {
        j: [(1, 23+j-1), 10, 'V', '0123456789', 'S'] for j in range(1, 15)
    }
    card_form = {
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 37,
            'mark_row_number': 14,
            'mark_valid_area_col_start': 23,
            'mark_valid_area_col_end': 36,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 13},
        'group_format': group1
    }
    return card_form


def omr_form2():
    omr_image_location = card2_location
    omr_image_list = [s for s in glob.glob(omr_image_location) if 'OMR' in s]
    group2 = {i + j*5: [(i, 2 + j*6), 4, 'H', 'ABCD', 'S'] for i in range(1, 6)
              for j in range(5)}
    card_form = {
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 31,
            'mark_row_number': 6,
            'mark_valid_area_col_start': 1,
            'mark_valid_area_col_end': 30,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 5
            },
        'group_format': group2
    }
    return card_form


def omr_form3():
    omr_image_location = card2_location
    omr_image_list = [s for s in glob.glob(omr_image_location) if 'Oomr' in s]
    group3 = {i: [(1, i), 10, 'V', '0123456789', 'S'] for i in range(1, 20)}
    card_form = {
        'image_file_list': omr_image_list,
        'mark_format': {  # 20, 11, 1, 19, 1, 10
            'mark_col_number': 20,
            'mark_row_number': 11,
            'mark_valid_area_col_start': 1,
            'mark_valid_area_col_end': 19,
            'mark_valid_area_row_start': 1,
            'mark_valid_area_row_end': 10
            },
        'group_format': group3
    }
    return card_form


def omr_form101():
    omr_image_location = card10x_location
    omr_image_list = glob.glob(omr_image_location)
    for fs in filter101:
        omr_image_list = [s for s in omr_image_list if fs in s]
    group_dict = {i-2: [(i, 3), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)}
    group_dict.update({i+5-2: [(i, 9), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group_dict.update({i+10-2: [(i, 15), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group_dict.update({i+15-2: [(i, 21), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group = group_dict
    card_form = {
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 25,
            'mark_row_number': 8,
            'mark_valid_area_col_start': 2,
            'mark_valid_area_col_end': 24,
            'mark_valid_area_row_start': 3,
            'mark_valid_area_row_end': 7
        },
        'group_format': group
    }
    return card_form


def omr_form102():
    omr_image_location = card10x_location
    omr_image_list = glob.glob(omr_image_location)
    for fs in filter102:
        omr_image_list = [s for s in omr_image_list if fs in s]
    group_dict = {i-2: [(i, 3), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)}
    group_dict.update({i+5-2: [(i, 9), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group_dict.update({i+10-2: [(i, 15), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group_dict.update({i+15-2: [(i, 21), 4, 'H', 'ABCD', 'S'] for i in range(3, 8)})
    group = group_dict
    card_form = {
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 25,
            'mark_row_number': 8,
            'mark_valid_area_col_start': 2,
            'mark_valid_area_col_end': 24,
            'mark_valid_area_row_start': 3,
            'mark_valid_area_row_end': 7
        },
        'group_format': group
    }
    return card_form
