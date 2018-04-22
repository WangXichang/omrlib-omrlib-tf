
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import omrlib as ol
#import form_test as ts
import pandas as pd

import weebar as wb


# f8 = ts.form_8()


def test(file_list,
         filenum=100,
         clip_top=0, clip_bottom=0, clip_right=0, clip_left=0,
         win=5,
         scan_scope=12,
         display=False
         ):
    if type(file_list) == str:
        files = [file_list]
    elif isinstance(file_list, ol.Former):
        files = file_list.file_list[0:filenum]
    elif isinstance(file_list, list):
        files = file_list
    else:
        print('form is not valid type')
        return
    st = time.time()
    bar=wb.BarcodeReader128()
    bar.image_scan_line_sum = win
    bar.image_scan_scope = scan_scope
    bar.set_image_clip(clip_bottom=clip_bottom,
                       clip_right=clip_right,
                       clip_left=clip_left,
                       clip_top=clip_top)
    # bar.set_image_files(file_list)
    # bar.code_num = code_num

    bar.get_barcode(file_list=files, display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time()-st, (time.time()-st)/len(files)))
    return bar
