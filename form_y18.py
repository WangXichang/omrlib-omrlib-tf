# *_* utf-8 *_*

import glob
# import numpy as np


def form_y18_101():
    omr_location = [r"d:/work/data/y18/101/*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')
    group = {
        j+6*h: [(41+j, 4+6*h), 4, 'H', 'ABCD', 'S'] for j in range(1, 7)
        for h in range(0, 5)
        }
    group.update({
        j + 30: [(41+j, 34), 4, 'H', 'ABCD', 'S'] for j in range(1, 4)
        })
    group.update({
        j+33: [(12, 23+j), 10, 'V', '0123456789', 'S'] for j in range(1, 16)
        })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 38,
            'mark_row_number': 50,
            'mark_valid_area_col_start': 4,
            'mark_valid_area_col_end': 38,
            'mark_valid_area_row_start': 11,
            'mark_valid_area_row_end': 49,
            'mark_location_block_row': 50,
            'mark_location_block_col': 1
        },
        'group_format': group,
        'image_clip': {
            'do_clip': False,
            'x_start': 0,
            'x_end': -1,
            'y_start': 0,
            'y_end': -1
        }
    }
    return card_form


def form_y18_109():
    omr_location = [r"d:/work/data/y18/109/*"
                    ]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')
    group = ({
        j: [(14, 23+j), 10, 'V', '0123456789', 'S'] for j in range(1, 16)
        })
    group.update({
        j+5*h+15: [(42+j+h*6, 5), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 3)
        })
    group.update({
        j+15+5*h+15: [(42+j+h*6, 14), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 3)
        })
    group.update({
        j+15+5*h+30: [(42+j+h*6, 23), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 3)
        })
    group.update({
        j+15+5*h+45: [(42+j+h*6, 32), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 2)
        })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 38,
            'mark_row_number': 61,
            'mark_valid_area_col_start': 4,
            'mark_valid_area_col_end': 38,
            'mark_valid_area_row_start': 14,
            'mark_valid_area_row_end': 60,
            'mark_location_block_row': 61,
            'mark_location_block_col': 1
        },
        'group_format': group,
        'image_clip': {
            'do_clip': False,
            'x_start': 0,
            'x_end': -1,
            'y_start': 0,
            'y_end': -1
        }
    }
    return card_form
