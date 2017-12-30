5# -*- utf-8 -*-


import glob


def form_17cr_S():
    omr_location = [
        r"d:\pythontest\omrphototest\S\omr\S86261\*",
        r"d:\pythontest\omrphototest\S\omr\S86262\*",
        r"d:\pythontest\omrphototest\S\omr\S86263\*",
        r"d:\pythontest\omrphototest\S\omr\S86268\*"]
    omr_location = ['d:/work/omr_test_data']
    omr_location = ['d:/work/data/somr']
    omr_image_list = []
    for loc in omr_location:
        loc1 = glob.glob(loc)
        for ds in loc1:
            omr_image_list = omr_image_list + \
                             glob.glob(ds + r'\*.jpg')

    group = {
        j + 5 * h: [(j+1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    }
    group.update({
        j + 20 + 5 * h: [(j + 6, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 40 + 5 * h: [(j + 11, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 60 + 5 * h: [(j + 15+1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 80 + 5 * h: [(j + 20 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })
    group.update({
        j + 84 + 5 * h: [(j + 24 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 6)
        for h in range(0, 4)
    })
    group.update({
        j + 104 + 5 * h: [(j + 29 + 1, 3 + 6 * h), 5, 'H', 'ABCDE', 'S'] for j in range(1, 5)
        for h in range(0, 1)
    })

    group.update({
        j + 108 + 5 * h: [(j + 33 + 1, 3 + 9 * h), 5, 'H', 'ABCDE', 'M'] for j in range(1, 6)
        for h in range(0, 2)
    })

    group.update({
        j + 118: [(j + 33 + 1, 21), 5, 'H', 'ABCDE', 'M'] for j in range(1, 3)
    })

    card_form = {
        'len': omr_image_list.__len__(),
        'image_file_list': omr_image_list,
        'mark_format': {
            'mark_col_number': 26,
            'mark_row_number': 40,
            'mark_valid_area_col_start': 3,
            'mark_valid_area_col_end': 25,
            'mark_valid_area_row_start': 2,
            'mark_valid_area_row_end': 39},
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 0,
            'x_end': -10,
            'y_start': 330,
            'y_end': -1
        }
    }
    return card_form
