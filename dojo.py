import pandas as pd
import numpy as np
#import lightgbm as lgb
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import check_cv
y = np.array([[1],[2]])


cv = check_cv(3, y, classifier=False)
print(y)
print(cv)
#print(b)
#print(len(b))
#
#a = np.zeros(len(b))
#print("a is ", a)

#result = np.sort(np.unique(b))
#print(result)

