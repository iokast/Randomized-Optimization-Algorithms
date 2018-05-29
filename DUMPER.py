# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

seg = pd.read_hdf('datasets.hdf','segmentation')     
segX = seg.drop('classification',1).copy().values
segY = seg['classification'].copy().values
le = preprocessing.LabelEncoder()
segY = le.fit_transform(segY)

# split dataset into training (70%) and test (30%) sets    
seg_trgX, seg_tstX, seg_trgY, seg_tstY = ms.train_test_split(segX, segY, test_size=0.3, random_state=0,stratify=segY)   
seg_trgY = np.atleast_2d(seg_trgY).T
seg_tstY = np.atleast_2d(seg_tstY).T

seg_trgX, seg_valX, seg_trgY, seg_valY = ms.train_test_split(seg_trgX, seg_trgY, test_size=0.2, random_state=1,stratify=seg_trgY)  

lb = preprocessing.LabelBinarizer()
seg_trgY = lb.fit_transform(seg_trgY)
seg_tstY = lb.fit_transform(seg_tstY)
seg_valY = lb.fit_transform(seg_valY)



tst = pd.DataFrame(np.hstack((seg_tstX,seg_tstY)))
trg = pd.DataFrame(np.hstack((seg_trgX,seg_trgY)))
val = pd.DataFrame(np.hstack((seg_valX,seg_valY)))
tst.to_csv('s_test.csv',index=False,header=False)
trg.to_csv('s_trg.csv',index=False,header=False)
val.to_csv('s_val.csv',index=False,header=False)