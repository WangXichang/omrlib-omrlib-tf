
import scipy.stats as stt
import omrlib as ol
import form_test as test

f1 = test.form_1()
f2 = test.form_21()
f3 = test.form_22()

def eva(file):
    rc = ol.read_check(file, disp_fig=1)
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
