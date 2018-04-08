
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import omrlib as ol
import form_test as ts

import weebar as wb


# f8 = ts.form_8()


def test(form, scan_scope=12, filenum=100,
         clip_top=0, clip_bottom=0, clip_right=0, clip_left=0,
         code_num=6, win=5,
         disp=False
         ):
    file_list = form.file_list[0:filenum]
    st = time.time()
    bar=wb.Barcoder()
    bar.image_detect_win_high = win
    bar.image_scan_scope = scan_scope
    bar.set_image_clip(clip_bottom=clip_bottom,
                       clip_right=clip_right,
                       clip_left=clip_left,
                       clip_top=clip_top)
    bar.set_image_files(file_list)
    bar.code_num = code_num
    bar.get_barcode('128', disp=disp)
    print('total time:', time.time()-st)
    return bar
