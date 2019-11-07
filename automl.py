"""model"""
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from util import log
from preprocess import sample
import time 
import random

class AutoSSLClassifier:
    def __init__(self, start_time, time_budget):
        self.model = None
        self.hypermodel = None
        self.start_time = start_time 
        self.time_budget = time_budget
        self.feature = None
    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }
        
        X_label, y_label, X_unlabeled, y_unlabeled = self._split_by_label(X, y)
           
        hyperparams = self._hyperopt(X_label, y_label, params)
        
        importance = self.hypermodel.feature_importance(importance_type='split')
        feature_name = self.hypermodel.feature_name()
        feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by='importance')
        
        self.feature = self.__identify_zero_importance()
        
        X_unlabeled = X_unlabeled.ix[:,self.feature]
        X_label = X_label.ix[:,self.feature]
        
        new_hyperparams = self._hyperopt(X_label, y_label, params)
        
        X_train, X_val, y_train, y_val = train_test_split(X_label, y_label, test_size=0.1, random_state=0)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train({**params, **new_hyperparams}, train_data, 300, valid_data, early_stopping_rounds=20, verbose_eval=50)
          
        return self

    def predict(self, X):
        X = X.ix[:,self.feature]
        return self.model.predict(X)
    
    def __identify_zero_importance(self):
        score = self.hypermodel.feature_importance()/self.hypermodel.feature_importance().sum()
        return list(np.where(score > np.percentile(score, 60))[0])
   
    def _split_by_label(self, X, y):
        y_label = pd.concat([y[y == -1], y[y == 1]])
        X_label = X.loc[y_label.index, :]
        y_unlabeled = y[y == 0]
        X_unlabeled = X.loc[y_unlabeled.index, :]
        return X_label, y_label, X_unlabeled, y_unlabeled

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            self.hypermodel = lgb.train({**params, **hyperparams}, train_data, 300,
                              valid_data, early_stopping_rounds=30, verbose_eval=0)
            
            score = self.hypermodel.best_score["valid_0"][params["metric"]]
            
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoPUClassifier:
    def __init__(self, start_time, time_budget):
        self.iter = 3
        self.models = []
        self.hypermodel = None
        self.feature = None
        self.start_time = start_time 
        self.time_budget = time_budget
        
    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        x_sample, y_sample = self._negative_sample(X, y)
        
        hyperparams = self._hyperopt(x_sample, y_sample, params)
        
        X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(x_sample, y_sample, test_size=0.2, random_state=1)

        train_data_temp = lgb.Dataset(X_train_temp, label=y_train_temp)
        valid_data_temp = lgb.Dataset(X_val_temp, label=y_val_temp)
		
        self.hypermodel = lgb.train({**params, **hyperparams}, train_data_temp, 400, valid_data_temp, early_stopping_rounds=30, verbose_eval=100)
		
        importance = self.hypermodel.feature_importance(importance_type='split')
        feature_name = self.hypermodel.feature_name()
        feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance} ).sort_values(by='importance')
            
        self.feature = self.__identify_zero_importance()
        
        x_sample = x_sample.ix[:,self.feature]
            
        new_hyperparams = self._hyperopt(x_sample, y_sample, params)
        
       
        for _ in range(self.iter):
            
            remain_time = self.time_budget - (time.time() - self.start_time)
            log(f"Remain time: {self.time_budget - (time.time() - self.start_time)}")
            
            if(remain_time/self.time_budget<=0.2):
                break
                      
            x_sample, y_sample = self._negative_sample(X, y)            
            x_sample = x_sample.ix[:,self.feature]
            
            X_train, X_val, y_train, y_val = train_test_split(x_sample, y_sample, test_size=0.2, random_state=1)

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train({**params, **new_hyperparams}, train_data, 500,
                              valid_data, early_stopping_rounds=30, verbose_eval=100)
            self.models.append(model)

        return self

    def predict(self, X):       
        X = X.ix[:,self.feature]
        for idx, model in enumerate(self.models):
        
            p = model.predict(X)
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

        return np.mean(prediction, axis=0)
    
    def __identify_zero_importance(self):
        score = self.hypermodel.feature_importance()/self.hypermodel.feature_importance().sum()
        return list(np.where(score > np.percentile(score, 50))[0])
    
    def _negative_sample(self, X, y):
        y_n_cnt, y_p_cnt = y.value_counts()
        y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
        
        y_sample = pd.concat([y[y == 0].sample(y_n_sample,  random_state=random.sample(range(0,2000),1)[0]), y[y == 1]])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample

    def _hyperopt(self, X, y, params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
            "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data, early_stopping_rounds=30, verbose_eval=0)
            
            score = model.best_score["valid_0"][params["metric"]]
         
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))
        hyperparams = space_eval(space, best)

        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoNoisyClassifier:
    def __init__(self, start_time, time_budget):
        self.model = []
        self.start_time = start_time 
        self.time_budget = time_budget

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
            
        }
        num = int(len(X)*0.4)
        X_sample, y_sample = sample(X, y, num)
        hyperparams = self._hyperopt(X_sample, y_sample, params)
        
        for i in range(350):
            
            remain_time = self.time_budget - (time.time() - self.start_time)
            log(f"Remain time: {self.time_budget - (time.time() - self.start_time)}")
            
            if(remain_time/self.time_budget<=0.2):
                break
  
            X_sample, y_sample = sample(X, y, num)
            
            X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.3, random_state=random.sample(range(0,2000),1)[0])
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train({**params, **hyperparams}, train_data, num_boost_round=500, valid_sets=[train_data,valid_data],   early_stopping_rounds=10, verbose_eval=100)
      
            self.model.append(model)
               
        return self

    def predict(self, X):
        
        for idx, model in enumerate(self.model):
            p = model.predict(X)
           
            if idx == 0:
                prediction = p
            else:
                prediction = np.vstack((prediction, p))

        return np.mean(prediction, axis=0)
        
    def _hyperopt(self, X, y, params):

        train_data = lgb.Dataset(X, label=y)
         
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.5)),
            "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
            "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 20, dtype=int)),
            "feature_fraction": hp.quniform("feature_fraction", 0.6, 1.0, 0.1),
            "reg_alpha": hp.uniform("reg_alpha", 0, 2),
            "reg_lambda": hp.uniform("reg_lambda", 0, 2),
            "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0)
        }

        def objective(hyperparams):
  
            model = lgb.cv({**params, **hyperparams},train_data, 1000, nfold=5, early_stopping_rounds=50, verbose_eval=0)
            
            score = max(model['auc-mean'])
            
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials,
                    algo=tpe.suggest, max_evals=2, verbose=1,
                    rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
        return hyperparams
    