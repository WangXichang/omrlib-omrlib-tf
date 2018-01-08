# *_* utf-8 *_*

import glob
# import numpy as np
import omr_lib1 as ol1


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
            'mark_location_row_no': 50,
            'mark_location_col_no': 1
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
            'mark_location_row_no': 61,
            'mark_location_col_no': 1
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


def form_y18_201():
    omr_location = [r"d:/work/data/y18/201/*"]
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                glob.glob(ds + r'\*.jpg')
    h_mark_num = 55
    group = ({
        j: [(14, 23+j), 10, 'V', '0123456789', 'S'] for j in range(1, 16)
        })
    group.update({
        j+15: [(h_mark_num-12+j, 4), 4, 'H', 'ABCD', 'S'] for j in range(1, 11)
        })
    group.update({
        j+25: [(h_mark_num-12+j, 10), 4, 'H', 'ABCD', 'S'] for j in range(1, 11)
        })
    group.update({
        j+35: [(h_mark_num-12+j, 18), 4, 'H', 'ABCD', 'S'] for j in range(1, 11)
        })
    group.update({
        j+45: [(h_mark_num-12+j, 24), 4, 'H', 'ABCD', 'S'] for j in range(1, 11)
        })
    group.update({
        j+55: [(h_mark_num-12+j, 31), 7, 'H', 'ABCDEFG', 'S'] for j in range(1, 6)
        })
    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 38,
            'mark_row_number': 55,
            'mark_valid_area_col_start': 4,
            'mark_valid_area_col_end': 38,
            'mark_valid_area_row_start': 14,
            'mark_valid_area_row_end': 54,
            'mark_location_row_no': 55,
            'mark_location_col_no': 1
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


def get_form_18y201():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/201/',
                          suffix='jpg')

    omrform.set_mark_format(row_number=55,
                            col_number=38,
                            valid_area_row_start=14,
                            valid_area_row_end=54,
                            valid_area_col_start=4,
                            valid_area_col_end=38,
                            location_row_no=55,
                            location_col_no=1
                            )

    omrform.set_image_clip(clip_x_start=0,
                           clip_x_end=-1,
                           clip_y_start=0,
                           clip_y_end=-1,
                           do_clip=False)

    for g in range(1, 16):  # 1-15
        omrform.set_group(group=g,
                          coord=(14, 23+g),
                          dire='V',
                          leng=10,
                          code='0123456789',
                          mode='S')
    for g in range(1, 11):  # 16-25
        omrform.set_group(group=g+15,
                          coord=(55-12, 4),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 26-35
        omrform.set_group(group=g+25,
                          coord=(55-12+g, 10),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 36-45
        omrform.set_group(group=g+35,
                          coord=(55-12+g, 18),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 46-55
        omrform.set_group(group=g+45,
                          coord=(55-12+g, 24),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 6):  # 56-60
        omrform.set_group(group=g+55,
                          coord=(55-12+g, 31),
                          dire='H',
                          leng=4,
                          code='ABCDEFG',
                          mode='S')

    omrform.get_form()
    return omrform


def get_form_18y202():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/202/',
                          suffix='jpg')

    omrform.set_mark_format(row_number=51,
                            col_number=38,
                            valid_area_row_start=13,
                            valid_area_row_end=50,
                            valid_area_col_start=5,
                            valid_area_col_end=38,
                            location_row_no=51,
                            location_col_no=1
                            )

    omrform.set_image_clip(clip_x_start=0,
                           clip_x_end=-1,
                           clip_y_start=0,
                           clip_y_end=-1,
                           do_clip=False)

    for g in range(1, 16):  # 1-15
        omrform.set_group(group=g,
                          coord=(13, 23+g),
                          dire='V',
                          leng=10,
                          code='0123456789',
                          mode='S')
    for g in range(1, 11):  # 16-25
        omrform.set_group(group=g+15,
                          coord=(39+g, 5),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 26-35
        omrform.set_group(group=g+25,
                          coord=(39+g, 14),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 36-45
        omrform.set_group(group=g+35,
                          coord=(39+g, 23),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    for g in range(1, 11):  # 46-55
        omrform.set_group(group=g+45,
                          coord=(39+g, 32),
                          dire='H',
                          leng=4,
                          code='ABCD',
                          mode='S')
    omrform.get_form()
    return omrform
