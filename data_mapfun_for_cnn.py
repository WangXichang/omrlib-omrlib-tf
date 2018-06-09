# -*-utf8-*-

import openomr as opo
import pandas as pd
import form_test as ft


def get_data(former):
    data_len = 1000
    sep_char = ';'
    label_list = []
    data_list = []
    file_num = len(former.file_list)
    for i, f in enumerate(former.file_list):
        print('now: {} of {} at file={}'.format(i, file_num, f))
        mt = opo.read_test(former, f, display=False)
        for k in mt.pos_prj_log:
            if k in mt.pos_valid_prj_log:
                label_list.append(1)
            else:
                label_list.append(0)
            dmax = max(mt.pos_prj_log[k])
            dmin = min(mt.pos_prj_log[k])
            dgap = dmax - dmin if dmax > dmin else 1
            data = [float_str((x - dmin)/dgap, 1, 6) + sep_char for x in mt.pos_prj_log[k]]
            if len(data) >= data_len:
                data = data[:data_len]
            elif len(data) < data_len-1:
                data = data + ['0.000000'+sep_char]*(data_len-len(data)-1) + ['0.000000']
            else:
                data = data + ['0.000000']
            data_list.append(''.join(data))
    dfm = pd.DataFrame({'label': label_list, 'data': data_list})
    return dfm


def float_str(x, d1, d2):
    fs = '{:'+str(d1+d2+1)+'.'+str(d2)+'f}'
    return fs.format(x)
