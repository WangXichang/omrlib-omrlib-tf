# -*- utf8 -*-
import time
import weebar as wb


def read128(file_list,
            clip_top=0, clip_bottom=0, clip_right=0, clip_left=0,
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
    br.set_image_clip(clip_bottom=clip_bottom,
                      clip_right=clip_right,
                      clip_left=clip_left,
                      clip_top=clip_top)
    br.get_barcode(file_list=file_list, display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time() - st, (time.time()-st) / len(file_list)))
    return br
