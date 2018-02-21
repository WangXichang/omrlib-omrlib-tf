# *_* utf-8 *_*

import omrlib as omrlib


def form_1():
    # define former
    former = omrlib.Former()

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

    former = omrlib.Former()

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


def form_22():
    
    # define former
    former = omrlib.Former()
    
    # define image file
    former.set_file_list(
        path='d:/work/data/omrimage2/', 
        substr='OMR01.jpg'    # assign substr in filename+pathstr
        )
    
    # define mark location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
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
    # _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
    cluster_coord = [(1, 2), (1, 8),  (1, 14), (1, 20), (1, 26)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,                    # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,          # area location left_top = (row, col)
            area_direction='v',                        # area direction V:top to bottom, H:left to right
            group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',       # group code for painting block
            group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
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


def form_4a():
    # define former
    former = omrlib.Former()

    # define image file
    former.set_file_list(
        path='d:/study/dataset/omrimage4/',
        substr='A86'  # assign substr in filename+pathstr
    )

    # define mark location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=16,
        col_number=37,
        valid_area_row_start=1,
        valid_area_row_end=15,
        valid_area_col_start=1,
        valid_area_col_end=36,
        location_row_no=16,
        location_col_no=37
    )

    # define cluster, _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1 + j * 5, 5 + j * 5) for j in range(6)]
    cluster_coord = [(1, 3 + j * 6) for j in range(6)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
            group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
        )
    cluster_group = [(31 + j * 5, 35 + j * 5) for j in range(6)]
    cluster_coord = [(6, 3 + j * 6) for j in range(6)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
            group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
        )
    cluster_group = [(61 + j * 5, 65 + j * 5) for j in range(6)]
    cluster_coord = [(11, 3 + j * 6) for j in range(6)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
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


def form_4c():
    # define former
    former = omrlib.Former()

    # define image file
    former.set_file_list(
        path='d:/study/dataset/omrimage4/',
        substr='C86'  # assign substr in filename+pathstr
    )

    # define mark location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=16,
        col_number=37,
        valid_area_row_start=1,
        valid_area_row_end=15,
        valid_area_col_start=1,
        valid_area_col_end=36,
        location_row_no=16,
        location_col_no=37
    )

    # define cluster, _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1 + j * 5, 5 + j * 5) for j in range(6)]
    cluster_coord = [(1, 3 + j * 6) for j in range(6)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
            group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
        )
    cluster_group = [(31 + j * 5, 35 + j * 5) for j in range(6)]
    cluster_coord = [(6, 3 + j * 6) for j in range(6)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
            group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
        )
    cluster_group = [(61 + j * 5, 65 + j * 5) for j in range(3)]
    cluster_coord = [(11, 3 + j * 6) for j in range(3)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
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


def form_5():
    
    # define former
    former = omrlib.Former()
    
    # define image file
    former.set_file_list(
        path='d:/study/dataset/omrimage5/', 
        substr='jpg'    # assign substr in filename+pathstr
        )
    
    # define mark location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=29,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=3,
        valid_area_col_end=25,
        location_row_no=6,
        location_col_no=29
        )
    
    # define cluster, _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1, 5), (6, 10)]
    cluster_coord = [(1, 3), (1, 22)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,                    # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,          # area location left_top = (row, col)
            area_direction='v',                        # area direction V:top to bottom, H:left to right
            group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',       # group code for painting block
            group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
            )
    
    # define image clip setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=0,
        clip_top=140,
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


def form_4d():
    
    # define former
    former = omrlib.Former()
    
    # define image file
    former.set_file_list(
        path='d:/study/dataset/omrimage4/',
        substr='D31'    # assign substr in filename+pathstr
        )
    
    # define mark location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=25,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=3,
        valid_area_col_end=24,
        location_row_no=6,
        location_col_no=25
        )
    
    # define cluster, _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1 + 5*j, 5 + 5*j) for j in range(3)]
    cluster_coord = [(1, 3 + 6*j) for j in range(3) ]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,                    # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,          # area location left_top = (row, col)
            area_direction='v',                        # area direction V:top to bottom, H:left to right
            group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',       # group code for painting block
            group_mode='M'           # group mode 'M': multi_choice, 'S': single_choice
            )

    # define area
    former.set_area(
        area_group_min_max=(16, 16),  # area group from min=a to max=b (a, b)
        area_location_leftcol_toprow=(1, 21),  # area location left_top = (row, col)
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction from left to right
        group_code='ABCD',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define image clip setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=0,
        clip_top=330,
        clip_bottom=640
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


def form_4g():
    # define former
    former = omrlib.Former()

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

    # define image clip setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=0,
        clip_top=330,
        clip_bottom=630
        )

    # define location for checking mark
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)

    # define image files list
    former.set_file_list(
        path='d:/study/dataset/omrimage4/',
        substr='G'  # assign substr in path to filter
    )

    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=25,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=3,
        valid_area_col_end=24,
        location_row_no=6,
        location_col_no=25
    )

    # define code cluster, contianing many areas
    # _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1 + 5 * j, 5 + 5 * j) for j in range(4)]
    cluster_coord = [(1, 3 + j * 6) for j in range(4)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,  # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,  # area location left_top = (row, col)
            area_direction='v',  # area direction V:top to bottom, H:left to right
            group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',  # group code for painting block
            group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
        )

    return former


def form_4i():
    
    # define former
    former = omrlib.Former()
    
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
    
    # define image clip setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=0,
        clip_top=330,
        clip_bottom=650
        )

    # define location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define image files list
    former.set_file_list(
        path='d:/study/dataset/omrimage4/',
        substr='I318'    # assign substr in path to filter
        )
    
    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=6,
        col_number=31,
        valid_area_row_start=1,
        valid_area_row_end=5,
        valid_area_col_start=3,
        valid_area_col_end=30,
        location_row_no=6,
        location_col_no=31
        )

    # define code cluster, contianing many areas
    # _group: (min_no, max_no), _coord: (left_col, top_row)
    cluster_group = [(1 + 5*j, 5 + 5*j) for j in range(5)]
    cluster_coord = [(1, 3 + 6*j) for j in range(5)]
    for gno, loc in zip(cluster_group, cluster_coord):
        former.set_area(
            area_group_min_max=gno,                    # area group from min=a to max=b (a, b)
            area_location_leftcol_toprow=loc,          # area location left_top = (row, col)
            area_direction='v',                        # area direction V:top to bottom, H:left to right
            group_direction='h',     # group direction 'V','v': up to down, 'H','h': left to right
            group_code='ABCD',       # group code for painting block
            group_mode='S'           # group mode 'M': multi_choice, 'S': single_choice
            )
                
    return former


def form_6():
    
    # define former
    former = omrlib.Former()
    
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
    
    # define image clip setting
    former.set_clip(
        do_clip=True,
        clip_left=0,
        clip_right=20,
        clip_top=330,
        clip_bottom=0
        )

    # define location for checking mark 
    former.set_check_mark_from_bottom(True)
    former.set_check_mark_from_right(True)
    
    # define image files list
    former.set_file_list(
        path='d:/study/dataset/omrimage6/', 
        substr='S86'    # assign substr in path to filter
        )
    
    # define mark format: row/column number, valid area, location
    former.set_mark_format(
        row_number=40,
        col_number=26,
        valid_area_row_start=1,
        valid_area_row_end=39,
        valid_area_col_start=1,
        valid_area_col_end=25,
        location_row_no=40,
        location_col_no=26
        )

    # define cluster1
    former.set_cluster(
        cluster_group_list=[(1 + 5 * j, 5 + 5 * j) for j in range(4)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(2, 3 + 6 * j) for j in range(4)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster2
    former.set_cluster(
        cluster_group_list=[(21 + 5 * j, 25 + 5 * j) for j in range(4)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(7, 3 + 6 * j) for j in range(4)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster3
    former.set_cluster(
        cluster_group_list=[(41 + 5 * j, 45 + 5 * j) for j in range(4)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(12, 3 + 6 * j) for j in range(4)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster4
    former.set_cluster(
        cluster_group_list=[(61 + 5 * j, 65 + 5 * j) for j in range(4)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(17, 3 + 6 * j) for j in range(4)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster5
    former.set_cluster(
        cluster_group_list=[(81, 84)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(22, 3)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster6
    former.set_cluster(
        cluster_group_list=[(85 + 5 * j, 89 + 5 * j) for j in range(4)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(26, 3 + 6 * j) for j in range(4)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster7
    former.set_cluster(
        cluster_group_list=[(105, 108)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(31, 3)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='S'  # group mode 'M': multi_choice, 'S': single_choice
    )

    # define cluster8
    former.set_cluster(
        cluster_group_list=[(109 + 5*j, 113 +5*j) for j in range(3)],  # group scope (min_no, max_no) per area
        cluster_coord_list=[(35, 3 + 9*j) for j in range(3)],  # left_top coord per area
        area_direction='v',  # area direction V:top to bottom, H:left to right
        group_direction='h',  # group direction 'V','v': up to down, 'H','h': left to right
        group_code='ABCDE',  # group code for painting block
        group_mode='M'  # group mode 'M': multi_choice, 'S': single_choice
    )

    return former
