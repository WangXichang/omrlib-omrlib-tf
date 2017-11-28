# *_* utf-8 *_*

import omr_lib1 as ol
import glob


def omr_form():
    # define omr iamges file_list
    loc = '3-2'
    card1_location = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg' \
        if loc == 'surface' else \
        'F:\\studies\\juyunxia\\omrimage1\\*.jpg'
    omr_image_list = glob.glob(card1_location)

    # group: {no: [pos_start, len, 'H/V', code, 'S/M'], ... }
    group = {
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
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 5,
            'x_end': 1200,
            'y_start': 5,
            'y_end': 390}
    }
    return card_form


if __name__ == '__main__':
    r_df = ol.omr_read_batch(omr_form())
