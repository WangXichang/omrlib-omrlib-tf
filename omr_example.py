# *_* utf-8 *_*

import omr_lib1 as ol
import glob
# import numpy as np


def omr_example_1():
    omr_image_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg'
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
    df_rsult = ol.omr_read_batch(card_form)
    return df_rsult
