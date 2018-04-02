
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import omrlib as ol
import form_test as ts

import weebar as wb


# f8 = ts.form_8()


def test(form, th_shift=70, scan_scope=12, filenum=100,
         tclip_bottom=0, tclip_right=0, tclip_left=0,
         code_num=6):
    file_list = form.file_list[0:filenum]
    st = time.time()
    bar=wb.Barcoder()
    bar.image_threshold_shift = th_shift
    bar.image_scan_scope = scan_scope
    bar.set_image_clip(clip_bottom=tclip_bottom,
                       clip_right=tclip_right,
                       clip_left=tclip_left)
    bar.set_image_files(file_list)
    bar.code_num = code_num
    bar.get_barcode('128')
    print('total time:', time.time()-st)
    return bar
