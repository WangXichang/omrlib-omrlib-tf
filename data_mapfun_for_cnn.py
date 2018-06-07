# -*-utf8-*-

import openomr as opo
import pandas as pd
import form_test as ft

def get_data(former):
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
            data_list.append([x/dmax for x in mt.pos_prj_log[k]])
    dfm = pd.DataFrame({'label': label_list, 'data': data_list})
    return dfm
