# _*_ utf-8 _*_

import openomr


def form_yw():
    
    # define former
    former = openomr.Former()
    
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
        do_clip=True,
        clip_left=0,
        clip_right=800,
        clip_top=350,
        clip_bottom=650
        )

    # define location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define image files list
    former.set_file_list(
        path='f:/studies/data/omrimage8/', 
        substr_list='_01.jpg'    # assign substr_list in path to filter
        )
    
    # define mark format: row/column number, valid area, location
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
    
    # define cluster
    former.set_cluster(
        cluster_group_list=[(1, 5), (6, 10), (11, 12)],    # group scope (min_no, max_no) per area
        cluster_coord_list=[(1, 3 + 6*j) for j in range(3)],    # left_top coord per area
        area_direction='v',      # area direction V:top to bottom, H:left to right
        group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCD',       # group code for painting block
        group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
        )
    
    # define score_dict
    # {group:{code:score,}}
    former.set_score(
        do_score = False,
        score_dict={
            1: {'A': 1},
            2: {'A': 1},
            3: {'A': 1},
            4: {'A': 1},
            5: {'A': 1},
            6: {'A': 1},
            7: {'A': 1},
            8: {'A': 1},
            9: {'A': 1},
            10: {'A': 1},
            11: {'A': 1},
            12: {'A': 1}
        }
    )
    
    return former


def form_ws_dell():
    # define former
    former = openomr.Former()

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
        do_clip=True,
        clip_left=0,
        clip_right=720,
        clip_top=300,
        clip_bottom=530
    )

    # define location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define image files list
    former.set_file_list(
        path='d:/work/data/g17/ws/',
        substr_list='jpg'  # assign substr in path to filter
    )

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=26,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=1,
        valid_area_col_end=25,
        location_row_no=6,
        location_col_no=26
    )

    # define cluster
    former.set_cluster(
        cluster_group_list=[(1, 5), (6, 10)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(1, 3), (1, 9)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCD',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define score_dict
    # {group:{code:score,}}
    former.set_score(
        do_score=False,
        score_dict={
            1: {'A': 1},
            2: {'A': 1},
            3: {'A': 1},
            4: {'A': 1},
            5: {'A': 1},
        }
    )

    return former


def form_ws_off():
    # define former
    former = openomr.Former()

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
    '''
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=680,
        clip_top=300,
        clip_bottom=530
    )'''
    former.set_clip_box(
        do_clip=True,
        clip_box_left=30,
        clip_box_top=330,
        clip_box_right=760,
        clip_box_bottom=490
    )
    # define location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define image files list
    former.set_file_list(
        path='f:/studies/data/data/wss/',
        substr_list='jpg'  # assign substr in path to filter
    )

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=26,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=1,
        valid_area_col_end=25,
        location_row_no=6,
        location_col_no=26
    )

    # define cluster
    former.set_cluster(
        cluster_group_list=[(1, 5), (6, 10)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(1, 3), (1, 9)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCD',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define score_dict
    # {group:{code:score,}}
    former.set_score(
        do_score=False,
        score_dict={
            1: {'A': 1},
            2: {'A': 1},
            3: {'A': 1},
            4: {'A': 1},
            5: {'A': 1},
        }
    )

    return former


def form_wz_off():
    
    # define former
    former = openomr.Former()
    
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
        do_clip=True,
        clip_left=0,
        clip_right=735,
        clip_top=300,
        clip_bottom=460
        )

    # define location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define image files list
    former.set_file_list(
        path='f:/studies/data/data/wzs/',
        substr_list='jpg'    # assign substr in path to filter
        )
    
    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=11,
        col_number=31,
        valid_area_row_start=1,
        valid_area_row_end=10,
        valid_area_col_start=1,
        valid_area_col_end=30,
        location_row_no=11,
        location_col_no=31
        )

    # define cluster1
    former.set_cluster(
        cluster_group_list=[(1 + j * 5, 5 + j * 5) for j in range(5)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(1, 3 + j * 6) for j in range(5)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCD',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster2
    former.set_cluster(
        cluster_group_list=[(26 + j*5, 30 + j*5) for j in range(2)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(6, 3 + j*6) for j in range(2)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCD',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define score_dict
    # {group:{code:score,}}
    former.set_score(
        do_score=False,
        score_dict={
            1: {'A': 1},
            2: {'A': 1},
            3: {'A': 1},
            4: {'A': 1},
            5: {'A': 1},
        }
    )
    
    return former


def form_xxx():
    
    # define former
    former = openomr.Former()
    
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
    #     left_top_coner: left(x,  column), top(y, row), 
    # right_bottom_coner: right(x, column), bottom(y, row)
    former.set_clip_box(
        do_clip=True,
        clip_box_left=0,
        clip_box_top=300,
        clip_box_right=760,
        clip_box_bottom=490
        )

    # define location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define image files list
    former.set_file_list(
        path='f:/studies/data/data/wss/', 
        substr_list='jpg'    # assign substr in path to filter
        )
    
    # define mark format: row/column number[1-n], valid area[1-n], location[1-n]
    former.set_mark_format(
        row_number=6,
        col_number=26,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=1,
        valid_area_col_end=25,
        location_row_no=6,
        location_col_no=26
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
            1: {'A': 1},
            2: {'A': 1},
            3: {'A': 1},
            4: {'A': 1},
            5: {'A': 1},
        }
    )
    
    return former
