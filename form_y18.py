# *_* utf-8 *_*

import glob
# import numpy as np
import omr_lib1 as ol1


def form_101():
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


def form_109():
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


def form_201():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/201/',
                          substr='jpg')

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
                          code_dire='V',
                          code_len=10,
                          code_str='0123456789',
                          code_mode='S')
    for g in range(1, 11):  # 16-25
        omrform.set_group(group=g+15,
                          coord=(43+g, 4),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 26-35
        omrform.set_group(group=g+25,
                          coord=(55-12+g, 10),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 36-45
        omrform.set_group(group=g+35,
                          coord=(55-12+g, 18),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 46-55
        omrform.set_group(group=g+45,
                          coord=(55-12+g, 24),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 6):  # 56-60
        omrform.set_group(group=g+55,
                          coord=(55-12+g, 31),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCDEFG',
                          code_mode='S')

    omrform.get_form()
    return omrform


def form_202():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/202/',
                          substr='jpg')

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
                          code_dire='V',
                          code_len=10,
                          code_str='0123456789',
                          code_mode='S')
    for g in range(1, 11):  # 16-25
        omrform.set_group(group=g+15,
                          coord=(39+g, 5),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 26-35
        omrform.set_group(group=g+25,
                          coord=(39+g, 14),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 36-45
        omrform.set_group(group=g+35,
                          coord=(39+g, 23),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    for g in range(1, 11):  # 46-55
        omrform.set_group(group=g+45,
                          coord=(39+g, 32),
                          code_dire='H',
                          code_len=4,
                          code_str='ABCD',
                          code_mode='S')
    omrform.get_form()
    return omrform


def form_203():
    omrform = ol1.OmrForm()
    omrform.set_file_list(path='d:/work/data/y18/203/',
                          substr='jpg')
    omrform.set_mark_format(row_number=51,
                            col_number=38,
                            valid_area_row_start=13,
                            valid_area_row_end=50,
                            valid_area_col_start=5,
                            valid_area_col_end=38,
                            location_row_no=51,
                            location_col_no=1
                            )
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(13, 24),
        area_v_move=0,
        area_h_move=1,
        code_dire='V',
        code_set='0123456789',
        code_mode='S'
        )
    col_pos = [5, 14, 23, 32]
    for no, col in enumerate(col_pos):
        omrform.set_group_area(
            area_group=(101 + 10 * no, 110 + 10 * no),
            area_loca=(40, col),
            area_v_move=1,
            area_h_move=0,
            code_dire='h',
            code_set='ABCD',
            code_mode='S')

    omrform.set_image_clip(clip_x_start=0,
                           clip_x_end=-1,
                           clip_y_start=0,
                           clip_y_end=-1,
                           do_clip=False)
    return omrform


def form_204():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/204/',
                          substr='jpg')

    omrform.set_mark_format(row_number=55,
                            col_number=38,
                            valid_area_row_start=13,
                            valid_area_row_end=53,
                            valid_area_col_start=4,
                            valid_area_col_end=38,
                            location_row_no=55,
                            location_col_no=1
                            )

    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(14, 24),
        area_v_move=0,
        area_h_move=1,
        code_dire='V',
        code_set='0123456789'
    )
    for i, col in enumerate([4, 10, 18, 24, 31]):
        omrform.set_group_area(
            area_group=(101 + i * 10, 110 + i * 10),
            area_loca=(44, col),
            area_v_move=1,
            area_h_move=0,
            code_dire='h',
            code_set='ABCD' if i < 4 else 'ABCDEFG'
        )
    omrform.set_image_clip(clip_x_start=0,
                           clip_x_end=-1,
                           clip_y_start=10,
                           clip_y_end=-1,
                           do_clip=True)

    return omrform


def form_311():
    omrform = ol1.OmrForm()

    omrform.set_file_list(path='d:/work/data/y18/311/',
                          substr='jpg')

    omrform.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=13,
        valid_area_row_end=48,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=50,
        location_col_no=1
        )

    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(13, 24),
        area_v_move=0,
        area_h_move=1,
        code_dire='V',
        code_set='0123456789'
    )
    for i, col in enumerate([5, 12, 19, 26, 34]):
        omrform.set_group_area(
            area_group=(101 + i * 10, 110 + i * 10),
            area_loca=(39, col),
            area_v_move=1,
            area_h_move=0,
            code_dire='h',
            code_set='ABCD'
        )
    omrform.set_image_clip(clip_x_start=0,
                           clip_x_end=-1,
                           clip_y_start=10,
                           clip_y_end=-1,
                           do_clip=False)

    return omrform


def form_314():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/314/723084/', substr='jpg')
    omrform.set_mark_format(
        row_number=44,
        col_number=26,
        valid_area_row_start=13,
        valid_area_row_end=43,
        valid_area_col_start=2,
        valid_area_col_end=26,
        location_row_no=44,
        location_col_no=1
        )
    # define a group_area for mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(13, 12),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define some group_areas (define multi group_area)
    g_area_lefttop_coord = [(41, 4), (42, 4)]
    g_area_no_min_max = [(101, 104), (105, 108)]
    for coord, gno in zip(g_area_lefttop_coord, g_area_no_min_max):
        # group: group_no = gno[0] to gno[1], left_top = coord
        omrform.set_group_area(
            area_group=(gno[0], gno[1]),
            area_loca=coord,
            area_v_move=0,   # area from top down to bottom
            area_h_move=6,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_315():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/315/738111/', substr='jpg')
    omrform.set_mark_format(
        row_number=56,
        col_number=26,
        valid_area_row_start=2,
        valid_area_row_end=55,
        valid_area_col_start=2,
        valid_area_col_end=26,
        location_row_no=1,
        location_col_no=1
        )
    # define a group_area for mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(13, 12),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster: many group_areas
    g_area_lefttop_coord = [(44, 5), (44, 13), (44, 21)]
    g_area_no_min_max = [(101, 110), (111, 120), (121, 130)]
    for coord, gno in zip(g_area_lefttop_coord, g_area_no_min_max):
        # group: group_no = gno[0] to gno[1], left_top = coord
        omrform.set_group_area(
            area_group=(gno[0], gno[1]),
            area_loca=coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_397():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/397', substr='jpg')
    omrform.set_mark_format(
        row_number=45,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=44,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=45,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*5, 105+i*5) for i in range(6)]
    cluster_area_coord = [(39, 4+i*6) for i in range(6)]
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        # group: group_no = g[0] to g[1], area left_top = coord
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD',     # group code for painting point
            code_mode='S' if group_min2max[0] < 121 else 'M'
        )
    return omrform


def form_398():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/398', substr='jpg')
    omrform.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=1,
        valid_area_row_end=49,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=50,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101 + i*10, 110+i*10) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(50-11, 5 + 7*i) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD',     # group code for painting point
            code_mode='S' if group_min2max[0] < 140 else 'M'
        )
    return omrform


def form_408():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/408', substr='jpg')
    omrform.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=49,
        valid_area_col_start=7,
        valid_area_col_end=38,
        location_row_no=50,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*10, 110+i*10) for i in range(4)]   # group no: (min, max)
    cluster_area_coord = [(39, 7+i*8) for i in range(4)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_414_omr01():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/414', substr='jpg')
    for f in omrform.form['image_file_list']:
        if 'Omr02' in f:
            omrform.form['image_file_list'].remove(f)
    omrform.set_mark_format(
        row_number=45,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=44,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=45,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*3, 103+i*3) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(41, 5+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_414_omr02():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/414', substr='Omr02.jpg')
    omrform.set_mark_format(
        row_number=45,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=30,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=1,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*3, 103+i*3) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(28, 5+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_498():
    omrform = ol1.OmrForm()
    omrform.set_image_clip(
       clip_x_start=1,
       clip_x_end=-1,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=False)
    omrform.set_file_list(path='d:/work/data/y18/498', substr='jpg')
    omrform.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=49,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=50,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    omrform.set_group_area(
        area_group=(1, 15),
        area_loca=(12, 24),
        area_v_move=0,   # area from top down to bottom
        area_h_move=1,   # area from left to right
        code_dire='v',  # group direction from left to right
        code_set='0123456789'   # group code for painting point
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*10, 110+i*10) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(39, 4+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        omrform.set_group_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_loca=area_coord,
            area_v_move=1,   # area from top down to bottom
            area_h_move=0,   # area from left to right
            code_dire='h',  # group direction from left to right
            code_set='ABCD'     # group code for painting point
        )
    return omrform


def form_497():
    f = form_397()
    f.set_file_list(path='d:/work/data/y18/497', substr='01.jpg')
    for gn in range(121, 131):
        f.set_group_mode(gn, 'M')
    return f
