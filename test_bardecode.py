# -*- utf8 -*-
import time
import weebar as wb


def test(file_list,
         clip_top=0, clip_bottom=0, clip_right=0, clip_left=0,
         win=5,
         scan_scope=12,
         display=False
         ):
    if type(file_list) == str:
        file_list = [file_list]
    elif isinstance(file_list, list):
        file_list = file_list
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

    bar.get_barcode(file_list=file_list, display=display)
    print('total time:{:5.2n},  mean time:{:4.2n}'.
          format(time.time() - st, (time.time()-st) / len(file_list)))
    return bar
