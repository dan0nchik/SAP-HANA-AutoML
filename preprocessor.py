import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Preprocessor:
    def __init__(self):
        # self.X_train = X_train
        # self.X_test = X_test
        # self.y_train = y_train
        # self.y_test = y_test
        pass

    def clean(self, df, params=None):
        # TODO
        # if df is not None:
        #     if df.isnull().sum().sum() != 0:
        #         for column in df:
        #             if 'int' in str(df[column].dtype):
        #                 imputer = SimpleImputer(fill_value=np.nan, strategy='mean')
        #                 df = imputer.fit_transform(df[column].values.reshape(-1, 1))
        return df

    def set_task(self, y):
        for col in y:
            unique = y[col].unique()
            # TODO make check smarter
            if unique[0] == 0 and unique[1] == 1:
                return [DecisionTreeClassifier()]
            else:
                return [LinearRegression()]
