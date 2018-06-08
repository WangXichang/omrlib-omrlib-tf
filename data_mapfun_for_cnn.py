# -*-utf8-*-

import openomr as opo
import pandas as pd
import form_test as ft

def get_data(former):
    data_len = 1000
    label_list = []
    data_list = []
    for f in former.file_list:
        print(f)
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
    dfm = pd.DataFrame({'label': label_list, 'data': data_list})
    return dfm
