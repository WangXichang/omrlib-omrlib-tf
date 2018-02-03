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


def train_kmeans(dfg, filename='model_kmean_xxx.m'):
    # dfg = get_group()
    clf = km(2)
    clf.fit(list(dfg['feat']))
    jb.dump(clf, filename)

