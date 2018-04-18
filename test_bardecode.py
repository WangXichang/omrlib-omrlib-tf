
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


def test(form,
         filenum=100,
         clip_top=0, clip_bottom=0, clip_right=0, clip_left=0,
         win=5,
         scan_scope=12,
         display=False
         ):
    if type(form) == str:
        file_list = [form]
    elif isinstance(form, ol.Former):
        file_list = form.file_list[0:filenum]
    elif isinstance(form, list):
        file_list = form
    else:
        print('form is not valid type')
        return
    st = time.time()
    bar=wb.Barcoder128()
    bar.image_detect_win_high = win
    bar.image_scan_scope = scan_scope
    bar.set_image_clip(clip_bottom=clip_bottom,
                       clip_right=clip_right,
                       clip_left=clip_left,
                       clip_top=clip_top)
    bar.set_image_files(file_list)
    # bar.code_num = code_num

    bar.get_barcode('128', display=display)
    print('total time:', time.time()-st)
    return bar
