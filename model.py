"""baseline"""
import os
import time
import pickle
import pandas as pd
from util import log, timeit
from automl import AutoSSLClassifier, AutoNoisyClassifier, AutoPUClassifier
from preprocess import clean_df, clean_table, feature_engineer
import numpy as np

class Model:
    """entry"""
    def __init__(self, info: dict):
        log(f"Info:\n {info}")

        self.model = None
        self.task = info['task']
        self.train_time_budget = info['time_budget']
        # self.pred_time_budget = info.get('pred_time_budget')
        self.cols_dtype = info['schema']

        self.dtype_cols = {'cat': [], 'num': [], 'time': []}

        for key, value in self.cols_dtype.items():
            if value == 'cat':
                self.dtype_cols['cat'].append(key)
            elif value == 'num':
                self.dtype_cols['num'].append(key)
            elif value == 'time':
                self.dtype_cols['time'].append(key)

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        """train model"""
        start_time = time.time()

        clean_table(X)
        clean_df(X)
        X = feature_engineer(X)
        
        log(f"Remain time: {self.train_time_budget - (time.time() - start_time)}")

        if self.task == 'ssl':
            self.model = AutoSSLClassifier(start_time, self.train_time_budget)
        elif self.task == 'pu':
            self.model = AutoPUClassifier(start_time, self.train_time_budget)
        elif self.task == 'noisy':
            self.model = AutoNoisyClassifier(start_time, self.train_time_budget)
        
        self.model.fit(X, y)

    @timeit
    def predict(self, X: pd.DataFrame):
        """predict"""
        start_time = time.time()

        clean_table(X)
        clean_df(X)
        X = feature_engineer(X)

        # log(f"Remain time: {self.pred_time_budget - (time.time() - start_time)}")

        prediction = self.model.predict(X)

        return pd.Series(prediction)

    @timeit
    def save(self, directory: str):
        """save model"""
        pickle.dump(
            self.model, open(os.path.join(directory, 'model.pkl'), 'wb'))

    @timeit
    def load(self, directory: str):
        """load model"""
        self.model = pickle.load(
            open(os.path.join(directory, 'model.pkl'), 'rb'))
    
    def _split_by_label(self, X, y):
        y_label = pd.concat([y[y == -1], y[y == 1]])
        X_label = X.loc[y_label.index, :]
        y_unlabeled = y[y == 0]
        X_unlabeled = X.loc[y_unlabeled.index, :]
        return X_label, y_label, X_unlabeled, y_unlabeled    
    