
import numpy as np
import pandas as pd
import scipy.stats as stt
import omrlib as ol
import form_test as test
import form_dict as fdict

f1 = test.form_1()
f21 = test.form_21()
f22 = test.form_22()
f41 = test.form_4a()
f42 = test.form_4c()
f43 = test.form_4d()
f44 = test.form_4g()
f45 = test.form_4i()
f5 = test.form_5()
f6 = test.form_6()
fall = [f1, f21, f22, f41, f42, f43, f44, f45, f5, f6]
# fall = [f41]

fall = list(fdict.tf.values()) + list(fdict.yf.values())

def mapfun_std():
    fname = []
    mapfuns = []
    feat_std_f = []
    feat_std_fp = []
    feat_std_g = []
    feat_var_p = []
    feat_label = []
    for f in fall:
        for file in f.file_list:
            print(file)
            rt = ol.read_test(f, file, disp_info=False)
            for hv, step in rt.pos_prj_log:
                mapf = rt.pos_prj_log[(hv, step)]
                mapfuns.append(mapf)
                valley = ol.Util.seek_valley_wid_from_mapfun(mapf)
                feat_std_f.append(np.std(mapf))
                feat_std_fp.append(np.std([y for y in mapf if y>=np.mean(mapf)]))
                feat_std_g.append(np.std(valley) if len(valley)>0 else 0)
                feat_var_p.append(rt.pos_peak_wid_var_log[(hv, step)] if (hv, step) in rt.pos_peak_wid_var_log else -1)
                if hv == 'h':
                    feat_label.append((1 if step in rt.pos_valid_hmapfun_std_log else 0))
                else:
                    feat_label.append((1 if step in rt.pos_valid_vmapfun_std_log else 0))
                fname.append(file.split('/')[-1])
    df = pd.DataFrame({'f': fname,
                       'mapfun': mapfuns,
                       'std_f': feat_std_f,
                       'std_fp': feat_std_fp,
                       'std_g': feat_std_g,
                       'var_p': feat_var_p,
                       'label': feat_label,
                       })
    return df


def eva(file):
    rc = ol.read_check(card_file=file, disp_check_result=1)
    log = rc.model.pos_start_end_list_log
    h_sels = {k:log[k] for k in log if k[0] == 'h'}
    v_sels = {k:log[k] for k in log if k[0] == 'v'}
    valid_hsels = {}
    valid_vsels = {}
    for k in h_sels:
        sel = rc.model.pos_start_end_list_log[k]
        if (len(sel[0]) == len(sel[1])) & (len(sel[0]) > 2):
            print('h mark count=', k[1])
            wids = [y - x for x, y in zip(sel[0], sel[1])]
            gap =[x-y for x, y in zip(sel[0][1:], sel[1][0:-1])]
            if len(wids)>0:
                print('\t', wids, '\n\t', stt.describe(wids).variance)
                print('\t', gap, '\n\t', stt.describe(gap).variance)
                valid_hsels.update({k:h_sels[k]})
            else:
                #print('\t---invalid list')
                continue

    for k in v_sels:
        sel = rc.model.pos_start_end_list_log[k]
        if (len(sel[0]) == len(sel[1])) & (len(sel[0]) > 2):
            print('v mark count=', k[1])
            wids = [y - x for x, y in zip(sel[0], sel[1])]
            gap =[x-y for x, y in zip(sel[0][1:], sel[1][0:-1])]
            if len(wids)>0:
                print('\t', wids, '\n\t', stt.describe(wids).variance)
                print('\t', gap, '\n\t', stt.describe(gap).variance)
                valid_vsels.update({k:v_sels[k]})
            else:
                #print('\t---invalid list')
                continue

    print('opt hmark count=', mapfun_opt(valid_hsels))
    print('opt vmark count=', mapfun_opt(valid_vsels))

    return rc


def mapfun_var(sel:list):    #start_end_list
    # sel = rc.model.pos_start_end_list_log[k]
    result = -1
    if (len(sel[0]) == len(sel[1])) & (len(sel[0]) > 2):
        wids = [y - x for x, y in zip(sel[0], sel[1])]
        gap = [x - y for x, y in zip(sel[0][1:], sel[1][0:-1])]
        if len(wids) > 0:
            # print('\t', wids, '\n\t', stt.describe(wids).variance)
            # print('\t', gap, '\n\t', stt.describe(gap).variance)
            result = 0.6*stt.describe(wids).variance + 0.4*stt.describe(gap).variance
    return result

def mapfun_opt(sels:dict):
    opt = {k:mapfun_var(sels[k]) for k in sels}
    mineval = min(opt.values())
    for k in opt:
        if opt[k] == mineval:
            return k
    return None
