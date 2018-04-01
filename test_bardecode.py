
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import omrlib as ol
import form_test as ts

import weebar as wb


f8 = ts.form_8()


def test(th_shift=70, scan_scope=12, filenum=100):
    file_list = f8.file_list[0:filenum]
    st = time.time()
    bar=wb.Barcoder()
    bar.image_threshold_shift = th_shift
    bar.image_scan_scope = scan_scope
    bar.set_image_clip(clip_bottom=600, clip_right=700)
    bar.set_image_files(file_list)
    bar.get_barcode('128')
    print('total time:', time.time()-st)
    return bar
