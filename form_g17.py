# _*_ utf-8 _*_

import omrlib


def form_yw():
    
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
        substr='_01.jpg'    # assign substr in path to filter
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
