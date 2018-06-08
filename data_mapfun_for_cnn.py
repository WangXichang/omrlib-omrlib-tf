# -*-utf8-*-

import openomr as opo
import pandas as pd
import form_test as ft


def get_data(former):
    data_len = 1000
    label_list = []
    data_list = []
    for i, f in enumerate(former.file_list):
        print(i, f)
        mt = opo.read_test(former, f, display=False)
        for k in mt.pos_prj_log:
            if k in mt.pos_valid_prj_log:
                label_list.append(1)
            else:
                label_list.append(0)
            dmax = max(mt.pos_prj_log[k])
            data = [x/dmax for x in mt.pos_prj_log[k]]
            if len(data) >= 1000:
                data = data[:1000]
            else:
                data = data + [0]*(1000-len(data))
            ds = ''.join([str(x)+'/' for x in data])
            ds = ds[:-1]
            data_list.append(ds)
            dmin = min(mt.pos_prj_log[k])
            dgap = dmax - dmin
            dgap = 1 if dgap == 0 else dgap
            dstr = ''.join([float_str((x-dmin)/(dgap), 1, 6)+'/' for x in mt.pos_prj_log[k]])
            data_list.append(dstr)
    dfm = pd.DataFrame({'label': label_list, 'data': data_list})
    return dfm


def float_str(x, d1, d2):
    fs = '{:'+str(d1+d2+1)+'.'+str(d2)+'f}'
    return fs.format(x)
