# *_* utf-8 *_*

import omrlib as omrlib


def form_101():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/101/783240/',
                         substr='jpg')
    # check mark setting
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=2,
        valid_area_row_end=50,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=1,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    former.set_area(
        area_group=(1, 15),     # group no from a to b (a, b)
        area_coord=(12, 24),     # group_area left_top_location = (row, col)
        area_direction='h',          # area from top down to bottom
        group_direction='v',          # group direction from left to right
        group_code='0123456789',  # group code for painting point
        group_mode='D'           # if <bool> else 'M'
    )
    # define cluster_area_group_code
    #   group no list: (min_no, max_no)
    cluster_area_group = [(101, 106), (107, 112), (113, 118), (119, 124), (125, 130), (131, 133)]
    #   area lt_corner: (left_col, top_row)
    cluster_area_coord = [(42, 4+i*6) for i in range(6)]
    for group_scope, loc_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=group_scope,
            area_coord=loc_coord,
            area_direction='v',      # area from top down to bottom
            group_direction='h',      # group direction from left to right
            group_code='ABCD',    # group code for painting point
            group_mode='S' if group_scope[0] < 113 else 'M'       # if group_min2max[0] in range(, ) else 'M','D'
        )
    return former


def form_109():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/109/', substr='jpg')
    # check mark setting
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
        row_number=61,
        col_number=38,
        valid_area_row_start=2,
        valid_area_row_end=61,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=61,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    former.set_area(
        area_group=(1, 15),     # group no from a to b (a, b)
        area_coord=(14, 24),     # group_area left_top_location = (row, col)
        area_direction='h',          # area from top down to bottom
        group_direction='v',          # group direction from left to right
        group_code='0123456789',  # group code for painting point
        group_mode='D'           # if <bool> else 'M'
    )
    # define cluster_area_group_code
    #   group no list: (min_no, max_no)
    cluster_area_group = [(101+5*n, 105+5*n) for n in range(11)]
    #   area lt_corner: (left_col, top_row)
    cluster_area_coord = []
    # 4 clusters
    for i in range(4):
        # 3 areas
        cluster_area_coord += [(43, 5 + i * 9)]
        cluster_area_coord += [(49, 5 + i * 9)]
        if i < 3:
            cluster_area_coord += [(55, 5 + i * 9)]
    for group_scope, loc_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=group_scope,
            area_coord=loc_coord,
            area_direction='v',      # area from top down to bottom
            group_direction='h',      # group direction from left to right
            group_code='ABCDE',    # group code for painting point
            group_mode='S'       # if group_min2max[0] in range(, ) else 'M','D'
        )
    return former


def form_201():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_file_list(path='d:/work/data/y18/201/783240/', substr='jpg')

    # check mark setting
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=55,
        col_number=38,
        valid_area_row_start=1,
        valid_area_row_end=54,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=55,
        location_col_no=1
    )

    # define area
    former.set_area(
        area_group=(1, 15),  # area group from min=a to max=b (a, b)
        area_coord=(14, 24),  # area location left_top = (row, col)
        area_direction='h',  # area direction V:top to bottom, H:left to right
        group_direction='v',  # group direction from left to right
        group_code='0123456789',  # group code set for encoding
        group_mode='D'  # 'M':multi_choice, 'S':single_choice
    )

    # define cluster
    # group for each area: (min_no, max_no)
    cluster_group = [(101 + i * 10, 110 + i * 10) for i in range(4)] + [(141, 145)]
    # location for each area: (left_col, top_row)
    cluster_coord = [(44, 4), (44, 10), (44, 18), (44, 24), (44, 31)]
    for group_scope, loc_coord in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group=group_scope,  # area group from min=a to max=b (a, b)
            area_coord=loc_coord,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction from left to right
            group_code='ABCD' if loc_coord[1] < 31 else 'ABCDEFG',  # group code set for encoding
            group_mode='S'  # 'M':multi_choice, 'S':single_choice
        )

    return former


def form_202():
    former = omrlib.Former()

    former.set_file_list(
        path='d:/work/data/y18/202/',
        substr='jpg')

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=51,
        col_number=38,
        valid_area_row_start=13,
        valid_area_row_end=50,
        valid_area_col_start=5,
        valid_area_col_end=38,
        location_row_no=51,
        location_col_no=1
        )

    # define area
    former.set_area(
        area_group=(1, 15),  # area group from min=a to max=b (a, b)
        area_coord=(13, 24),  # area location left_top = (row, col)
        area_direction='h',         # area direction V:top to bottom, H:left to right
        group_direction='v',              # group direction from left to right
        group_code='0123456789',      # group code set for encoding
        group_mode='B'               # 'M':multi_choice, 'S':single_choice
    )

    # define cluster
    # group for each area: (min_no, max_no)
    cluster_group = [(101 + i * 10, 110 + i * 10) for i in range(4)]
    # location for each area: (left_col, top_row)
    cluster_coord = [(40, 5+j*9) for j in range(4)]
    for group_scope, loc_coord in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group=group_scope,  # area group from min=a to max=b (a, b)
            area_coord=loc_coord,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',      # group direction from left to right
            group_code='ABCD',    # group code set for encoding
            group_mode='S'       # 'M':multi_choice, 'S':single_choice
        )

    return former


def form_203():
    former = omrlib.Former()
    former.set_file_list(path='d:/work/data/y18/203/',
                         substr='jpg')
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=51,
        col_number=38,
        valid_area_row_start=13,
        valid_area_row_end=50,
        valid_area_col_start=5,
        valid_area_col_end=38,
        location_row_no=51,
        location_col_no=1
        )
    former.set_area(
        area_group=(1, 15),
        area_coord=(13, 24),
        area_direction='h',
        group_direction='V',
        group_code='0123456789',
        group_mode='D'
        )
    col_pos = [5, 14, 23, 32]
    for no, col in enumerate(col_pos):
        former.set_area(
            area_group=(101 + 10 * no, 110 + 10 * no),
            area_coord=(40, col),
            area_direction='v',
            group_direction='h',
            group_code='ABCD',
            group_mode='S')
    return former


def form_204():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_file_list(path='d:/work/data/y18/204/',
                         substr='jpg')

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=55,
        col_number=38,
        valid_area_row_start=13,
        valid_area_row_end=53,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=55,
        location_col_no=1
        )

    former.set_area(
        area_group=(1, 15),
        area_coord=(14, 24),
        area_direction='h',
        group_direction='V',
        group_code='0123456789',
        group_mode='D'
        )
    for i, col in enumerate([4, 10, 18, 24]):
        former.set_area(
            area_group=(101 + i * 10, 110 + i * 10),
            area_coord=(44, col),
            area_direction='v',
            group_direction='h',
            group_code='ABCD'
        )
    former.set_area(
        area_group=(156, 160),
        area_coord=(44, 31),
        area_direction='v',
        group_direction='h',
        group_code='ABCDEFG'
        )

    return former


def form_311():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_file_list(path='d:/work/data/y18/311/',
                         substr='jpg')

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=50,
        col_number=38,
        valid_area_row_start=13,
        valid_area_row_end=48,
        valid_area_col_start=4,
        valid_area_col_end=38,
        location_row_no=50,
        location_col_no=1
        )

    former.set_area(
        area_group=(1, 15),
        area_coord=(13, 24),
        area_direction='h',
        group_direction='V',
        group_code='0123456789',
        group_mode='D'
    )
    for i, col in enumerate([5, 12, 19, 26, 34]):
        former.set_area(
            area_group=(101 + i * 10, 110 + i * 10),
            area_coord=(39, col),
            area_direction='v',
            group_direction='h',
            group_code='ABCD'
        )

    return former


def form_314():
    former = omrlib.Former()

    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/314/723084/',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(13, 12),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define some group_areas (define multi group_area)
    g_area_lefttop_coord = [(41, 4), (42, 4)]
    g_area_no_min_max = [(101, 104), (105, 108)]
    for coord, gno in zip(g_area_lefttop_coord, g_area_no_min_max):
        # group: group_no = gno[0] to gno[1], left_top = coord
        former.set_area(
            area_group=(gno[0], gno[1]),
            area_coord=coord,
            area_direction='h',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD'     # group code for painting point
        )
    return former


def form_315():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/315/738111/',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(13, 12),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster: many group_areas
    g_area_lefttop_coord = [(44, 5), (44, 13), (44, 21)]
    g_area_no_min_max = [(101, 110), (111, 120), (121, 130)]
    for coord, gno in zip(g_area_lefttop_coord, g_area_no_min_max):
        # group: group_no = gno[0] to gno[1], left_top = coord
        former.set_area(
            area_group=(gno[0], gno[1]),
            area_coord=coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD'     # group code for painting point
        )
    return former


def form_397():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/397',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*5, 105+i*5) for i in range(6)]
    cluster_area_coord = [(39, 4+i*6) for i in range(6)]
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        # group: group_no = g[0] to g[1], area left_top = coord
        former.set_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',
            group_code='ABCD',     # group code for painting point
            group_mode='S' if group_min2max[0] < 121 else 'M'
        )
    return former


def form_398():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/398',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101 + i*10, 110+i*10) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(50-11, 5 + 7*i) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_code='ABCD',     # group code for painting point
            group_mode='S' if group_min2max[0] < 140 else 'M'
        )
    return former


def form_408():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/408',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*10, 110+i*10) for i in range(4)]   # group no: (min, max)
    cluster_area_coord = [(39, 7+i*8) for i in range(4)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=group_min2max,
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD',     # group code for painting point
            group_mode='S'
        )
    return former


def form_414_omr01():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/414',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    for f in former.form['image_file_list']:
        if 'Omr02' in f:
            former.form['image_file_list'].remove(f)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*3, 103+i*3) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(41, 5+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD'     # group code for painting point
        )
    return former


def form_414_omr02():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/414',
                         substr='Omr02.jpg')
    former.set_check_mark_from_bottom(False)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
        row_number=45,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=32,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=1,
        location_col_no=1
        )
    # define group_area, including mulit_groups
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*3, 103+i*3) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(28, 5+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD'     # group code for painting point
        )

    return former


def form_498():
    former = omrlib.Former()
    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )
    former.set_file_list(path='d:/work/data/y18/498',
                         substr='jpg')
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)
    former.set_mark_format(
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
    former.set_area(
        area_group=(1, 15),
        area_coord=(12, 24),
        area_direction='h',   # area from top down to bottom
        group_direction='v',  # group direction from left to right
        group_code='0123456789',   # group code for painting point
        group_mode='D'
    )
    # define area_cluster, including multi group_areas
    cluster_area_group = [(101+i*10, 110+i*10) for i in range(5)]   # group no: (min, max)
    cluster_area_coord = [(39, 4+i*7) for i in range(5)]    # area coord: (left col no, top row no)
    for group_min2max, area_coord in zip(cluster_area_group, cluster_area_coord):
        former.set_area(
            area_group=(group_min2max[0], group_min2max[1]),
            area_coord=area_coord,
            area_direction='v',   # area from top down to bottom
            group_direction='h',  # group direction from left to right
            group_code='ABCD'     # group code for painting point
        )
    return former


def form_497():
    former = omrlib.Former()

    # clip image setting
    # clip image setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
        )

    former.set_file_list(path='d:/work/data/y18/497/726121/',
                         substr='jpg')

    # check mark setting
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(False)

    former.set_mark_format(
        row_number=45,
        col_number=38,
        valid_area_row_start=12,
        valid_area_row_end=44,
        valid_area_col_start=2,
        valid_area_col_end=38,
        location_row_no=45,
        location_col_no=1
    )

    # define area
    former.set_area(
        area_group=(1, 15),  # area group from min=a to max=b (a, b)
        area_coord=(12, 24),  # area location left_top = (row, col)
        area_direction='h',  # area direction V:top to bottom, H:left to right
        group_direction='v',  # group direction from left to right
        group_code='0123456789',  # group code set for encoding
        group_mode='D'  # 'M':multi_choice, 'S':single_choice
    )

    # define cluster
    # group for each area: (min_no, max_no)
    cluster_group = [(101+j*5, 105+j*5) for j in range(6)]
    # location for each area: (left_col, top_row)
    cluster_coord = [(39, 4+j*6) for j in range(6)]
    for group_scope, loc_coord in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group=group_scope,  # area group from min=a to max=b (a, b)
            area_coord=loc_coord,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction from left to right
            group_code='ABCD',  # group code set for encoding
            group_mode='S' if loc_coord[1] < 28 else 'M'  # 'M':multi_choice, 'S':single_choice
        )

    return former
