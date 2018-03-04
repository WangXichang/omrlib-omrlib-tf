# -*- utf8 -*-

import omrlib as ol
import form_test as test
import form_y18 as y18

"""
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
ft = {'f1': f1, 'f21': f21, 'f22': f22, 'f41': f41, 'f42': f42, 'f43': f43, 'f44': f44, 'f45': f45,
      'f5': f5, 'f6': f6}
"""

tf = {k: test.__dict__[k]() for k in test.__dict__.keys() if k[0:5] == 'form_'}
yf = {k: y18.__dict__[k]() for k in y18.__dict__.keys() if k[0:5] == 'form_'}
