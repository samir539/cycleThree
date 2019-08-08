import pandas as pd
import numpy as np
#import lightgbm as lgb
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
a = np.random.rand(3,3)
b = np.random.rand(3,3)
instance1 = TransformerMixin()

c = instance1.fit_transform(a, b )

print(c)