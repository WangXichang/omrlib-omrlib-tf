# *_* utf-8 *_*

import omr_lib1 as omrlib


def form_1():
    # define former
    former = omrlib.FormBuilder()

    # define image file
    former.set_file_list(
        path='d:/study/dataset/omrimage1/',
        substr='jpg'  # assign substr in filename+pathstr
    )

    # define mark location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=14,
        col_number=37,
        valid_area_row_start=1,
        valid_area_row_end=10,
        valid_area_col_start=23,
        valid_area_col_end=36,
        location_row_no=14,
        location_col_no=37
    )

    # define area
    former.set_area(
        area_group_min_max=(1, 14),  # area group from min=a to max=b (a, b)
        area_location_leftcol_toprow=(1, 23),  # area location left_top = (row, col)
        area_direction='h',  # area direction V:top to bottom, H:left to right
        group_direction='v',  # group direction from left to right
        group_code='0123456789',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define image clip setting
    former.set_clip(
        do_clip=False,
        clip_left=0,
        clip_right=0,
        clip_top=0,
        clip_bottom=0
    )

    # define model parameters
    former.set_model_para(
        valid_painting_gray_threshold=35,
        valid_peak_min_width=3,
        valid_peak_min_max_width_ratio=5,
        detect_mark_vertical_window=20,
        detect_mark_horizon_window=20,
        detect_mark_step_length=5,
        detect_mark_max_count=100
    )

    return former


def form_21():
    former = omrlib.FormBuilder()
    former.set_image_clip(
       clip_x_start=1,
       clip_x_end=-40,
       clip_y_start=1,
       clip_y_end=-1,
       do_clip=True)

    former.set_file_list(path='d:/work/data/omrimage2/',
                         substr='OMR01.jpg')

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    former.set_mark_format(
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
        former.set_area(
            area_group_min_max=group_min2max,
            area_location_leftcol_toprow=area_coord,
            area_direction='v',       # area from top down to bottom
            group_direction='h',      # group direction from left to right
            group_code='ABCD',    # group code for painting point
            group_mode='S'       # if group_min2max[0] in range(, ) else 'M'
        )
    return former


def form_22():

    former = omrlib.FormBuilder()

    former.set_file_list(path='d:/work/data/omrimage2/',
                         substr='omr01.jpg')

    # clip image setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=35,
        clip_top=0,
        clip_bottom=90
        )

    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    former.set_mark_format(
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
        former.set_area(
            area_group_min_max=group_min2max,
            area_location_leftcol_toprow=area_coord,
            area_direction='h',         # area from top down to bottom
            group_direction='v',        # group direction from left to right
            group_code='0123456789',    # group code for painting point
            group_mode='D'              # if group_min2max[0] in range(, ) else 'M'
        )
    return former
