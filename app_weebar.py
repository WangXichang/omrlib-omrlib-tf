# -*- utf8 -*-
import time
import weebar as wb


def read128(file_list,
            box_top=0, box_left=0, box_bottom=0, box_right=0,
            ratio_row=None, ratio_col=None,
            display=False
            ):
    docstrings = \
        '''    
        :param file_list: image files with barcode
        :param clip_top:  clip top rows of image
        :param clip_bottom: clip bottom rows of image 
        :param clip_right:  clip right columns of image
        :param clip_left:   clip left columns of image
        :param display: display messages in processing
        :return: model, object of BarcodeReader128
        '''
    if type(file_list) == str:
        file_list = [file_list]
    elif isinstance(file_list, list):
        file_list = file_list
    else:
        print('form is not valid type')
        return
    st = time.time()
    br=wb.BarcodeReader128()
    br.get_barcode(file_list=file_list,
                   ratio_row=ratio_row, ratio_col=ratio_col,
                   box_top=box_top, box_left=box_left, box_bottom=box_bottom, box_right=box_right,
                   display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time() - st, (time.time()-st) / len(file_list)))
    return br
