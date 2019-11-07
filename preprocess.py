import datetime
from sklearn.decomposition import TruncatedSVD
import CONSTANT
from util import log, timeit
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
AGGREGATE_TYPE = ['min', 'max', 'mean', 'median', 'var']

@timeit
def clean_table(table):
    
    clean_df(table)

@timeit
def clean_df(df):
    
    fillna(df)

@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]:
        df[c].fillna(df[c].max(), inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
        df[c].fillna("0", inplace=True)

    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

@timeit
def feature_engineer(df):
    num_col = [c for c in df if c.startswith(CONSTANT.NUMERICAL_PREFIX)]
    cat_col = [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]
    time_col = [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]
  
    if(len(cat_col)>1):
        crosscount(df, cat_col)
        
    transform_categorical_hash(df)
 
    transform_datetime(df)
    
    return df

def feature_count(df, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        # return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del df[new_feature]
    except:
        pass
    temp = df.groupby(features).size().reset_index().rename(columns={0: new_feature})
    df = df.merge(temp, 'left', on=features)
    print(df)
    return df
   
@timeit
def transform_datetime(df):
    for c in [c for c in df if c.startswith(CONSTANT.TIME_PREFIX)]:
       
        df['year_{}'.format(c)] = df[c].dt.year
        df['month_{}'.format(c)] = df[c].dt.month
        df['day_{}'.format(c)] = df[c].dt.day
        df['hour_{}'.format(c)] = df[c].dt.hour
        df['weekday_{}'.format(c)] = df[c].dt.dayofweek
        df['dayofyear_{}'.format(c)] = df[c].dt.dayofyear
        df['weekofyear_{}'.format(c)] = df[c].dt.weekofyear
        df['quarter_{}'.format(c)] = df[c].dt.quarter
        df['minute_{}'.format(c)] = df[c].dt.minute
        df['second_{}'.format(c)] = df[c].dt.second
     
        df.drop(c, axis=1, inplace=True)
        
@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CONSTANT.CATEGORY_PREFIX)]:
    
        df['count_{}'.format(c)] = df.groupby(c)[c].transform('count')
        
        lbl = preprocessing.LabelEncoder()
        df['labelencoder_{}'.format(c)] = lbl.fit_transform(df[c].astype(object))
        
        df.drop([c], axis=1, inplace=True)
       
@timeit
def sample(X, y, nrows):
    
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=random.sample(range(0,50),1)[0])
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample

def crosscount(df, col_list):
  
    assert isinstance(col_list, list)
    assert len(col_list) >= 2
    
    name = "crosscount_"+ '_'.join(col_list)
    df[name] = df.groupby(col_list)[col_list[0]].transform('count')
    