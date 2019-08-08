import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.base import TransformerMixin, BaseEstimator



class ClassifierTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
    
    def _get_labels(self, y):       # this appears to just return y_labels I have not a clue how this is useful
        y_labels = np.zeros(len(y))       #array of zeros of length y (across)  [0,0,0,0,0,0,0,0,0...]
        y_us = np.sort(np.unique(y))      #creates an array of the unique elements of y and then sorts in ascending order **
        step = int(len(y_us) / self.n_classes)    
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        X = X.replace([np.inf,-np.inf], np.nan)         # finds all infinte values and replaces with NaN
        X = X.fillna(0)                             # fills NaN with zero 
        y_labels = self._get_labels(y) 
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator)) # give the KFold variable KFold(n_splits=3, random_state=None, shuffle=False)
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            X = np.array(X)
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        X = X.replace([np.inf,-np.inf], np.nan)
        X = X.fillna(0)
        X = np.array(X)
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

all_features = ['type',  'atom_x', 'x_x', 'y_x','z_x', 'n_bonds_x', 'atom_y', 'x_y', 'y_y',
       'z_y', 'n_bonds_y', 'C', 'F', 'H', 'N', 'O', 'distance', 'dist_mean_x','dist_mean_y',
       'x_dist', 'y_dist', 'z_dist', 'x_dist_abs', 'y_dist_abs', 'z_dist_abs','inv_distance3']
cat_features = ['type','atom_x','atom_y']