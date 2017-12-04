# *_* utf-8 *_*

import omr_lib1 as ol
import glob


# define omr iamges file_list
loc_surface_omr1 = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg'
loc_surface_omr2 = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg'
loc_surface_omr3 = 'C:\\Users\\wangxichang\\students\\ju\\testdata\\omr1\\*.jpg'

loc_32_omr1 = 'F:\\studies\\juyunxia\\omrimage1\\*.jpg'
loc_32_omr2 = 'F:\\studies\\juyunxia\\omrimage2\\*.jpg'
loc_32_omr3 = 'F:\\studies\\juyunxia\\omrimage3\\*.jpg'


def get_omr_form(loc: str):
    omr_image_list = glob.glob(loc)
    if len(omr_image_list) == 0:
        print(f'no omr image file found in {loc}')
        return

    # group: {no: [pos_start(row,col), len, 'H/V', code, 'S/M'], ... }
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
            'mark_valid_area_row_end': 13
        },
        'group_format': group,
        'image_clip': {
            'do_clip': True,
            'x_start': 30,
            'x_end': 1200,
            'y_start': 5,
            'y_end': 390
        }
    }
    return card_form


def read_omr(form: dict):
    return ol.omr_read_batch(form)

def help():
    doc = """
    usage guide
    import omr_read_ex01 as ex
    locex = ex.loc_32_omr1
    formex = ex.get_omr_form(locex)
    omrdf = ex.read_omr(formex)
    """
    print(doc)
