# -*- coding: utf-8 -*-

import pandas as pd

segmentation = pd.read_csv('./segmentation.csv',header=None)
#le = LabelEncoder()
#segmentation = segmentation.values
#segmentation[:,0] = le.fit_transform(segmentation[:,0])
#segmentation = pd.DataFrame(segmentation)
segmentation.columns = ['classification','REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2',
                        'VEDGE-MEAN','VEDGE-SD','HEDGE-MEAN','HEDGE-SD','INTENSITY-MEAN','RAWRED-MEAN','RAWBLUE-MEAN','RAWGREEN-MEAN',
                        'EXRED-MEAN','EXBLUE-MEAN','EXGREEN-MEAN','VALUE-MEAN','SATURATION-MEAN','HUE-MEAN']
segmentation.to_hdf('datasets.hdf','segmentation',complib='blosc',complevel=9)

