# *_* utf-8 *_*

import omr_lib1 as ol
import glob
# import numpy as np


def form2_OMR01():
    omrform = ol.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/omrimage2/', substr='OMR01.jpg')
    omrform.set_mark_format(
        row_number=6,
        col_number=31,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=1,
        valid_area_col_end=30,
        location_row_no=6,
        location_col_no=31
        )
    # define area_cluster, including multi group_areas
    # group2 = {i + j * 5: [(i, 2 + j * 6), 4, 'H', 'ABCD', 'S'] for i in range(1, 6)
    cluster_area_group = [(j*5+1, j*5+5) for j in range(5)]   # group no list: (min, max)
    cluster_area_coord = [(1, 2+j*6) for j in range(5)]
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_area(
            area_group_min_max=group_min2max,
            area_location_leftcol_toptow=area_coord,
            area_direction=1,      # area from top down to bottom
            area_h_move=0,      # area from left to right
            code_dire='h',      # group direction from left to right
            code_set='ABCD',    # group code for painting point
            code_mode='S'       # if group_min2max[0] in range(, ) else 'M'
        )
    return omrform


def form2_omr01():
    omrform = ol.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/omrimage2/', substr='omr01.jpg')
    omrform.set_mark_format(
        row_number=11,
        col_number=20,
        valid_area_row_start=1,
        valid_area_row_end=10,
        valid_area_col_start=1,
        valid_area_col_end=19,
        location_row_no=11,
        location_col_no=20
        )
    # define area_cluster, including multi group_areas
    # group2 = {i + j * 5: [(i, 2 + j * 6), 4, 'H', 'ABCD', 'S'] for i in range(1, 6)
    cluster_area_group = [(1, 19)]   # group no list: (min, max)
    cluster_area_coord = [(1, 1)]
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_area(
            area_group_min_max=group_min2max,
            area_location_leftcol_toptow=area_coord,
            area_direction=0,      # area from top down to bottom
            area_h_move=1,      # area from left to right
            code_dire='V',      # group direction from left to right
            code_set='0123456789',    # group code for painting point
            code_mode='D'       # if group_min2max[0] in range(, ) else 'M'
        )
    return omrform

def omr_form3():
    loc = 'office'
    card2_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr2\\*.jpg' \
        if loc == 'surface' else \
        'F:\\studies\\juyunxia\\omrimage2\\*.jpg'
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
        'group_format': group3,
        'image_clip': {
            'do_clip': False,
            'x_start': 0,
            'x_end': 1,
            'y_start': 0,
            'y_end': 1}
    }
    return card_form


def omr_form101():
    loc = 'office'
    filter101 = ['1-']
    card10x_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr0\\*.jpg' \
                       if loc == 'surface' else \
                       'F:\\studies\\juyunxia\\omrimage2\\*.jpg'
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
        'group_format': group,
        'image_clip': {
            'do_clip': False,
            'x_start': 0,
            'x_end': 1,
            'y_start': 0,
            'y_end': 1}
    }
    return card_form


def omr_form102():
    loc = 'office'
    filter102 = ['2-']
    card10x_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr0\\*.jpg' \
        if loc == 'surface' else \
        'F:\\studies\\juyunxia\\omrimage2\\*.jpg'
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
        'group_format': group,
        'image_clip': {
            'do_clip': False,
            'x_start': 0,
            'x_end': 1,
            'y_start': 0,
            'y_end': 1}
    }
    return card_form
