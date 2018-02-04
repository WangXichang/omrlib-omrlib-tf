# -*- utf8 -*-

import omr_lib1 as ol
import form_test as ftt
from sklearn.externals import joblib as jb
from sklearn.cluster import KMeans as km
from sklearn import svm
import pandas as pd


f21=ftt.form2_omr01()
f22=ftt.form2_OMR01()

f21filter = [f21.form['image_file_list'][i] for i in [3, 107]]


def get_group(former, filterfile=()):
    dfgroup = None
    for f in former.form['image_file_list']:
        if f in filterfile:
            continue
        if dfgroup is None:
            dfgroup = ol.omr_test(former, f).omr_result_dataframe_groupinfo
        else:
            dfgroup = dfgroup.append(ol.omr_test(former, f).omr_result_dataframe_groupinfo)
    return dfgroup


def train_model(dfg, modelname='kmeans', savefilename=''):
    md = ol.SklearnModel()
    md.set_data(data_feat=list(dfg['feat']),
                data_label=list(dfg['label']))
    md.classify_number = 2
    md.make_model(modelname)
    if len(savefilename) > 0:
        jb.dump(md.model, savefilename)
