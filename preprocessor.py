import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from impsettings import ImputerSettings


class Preprocessor:
    def __init__(self, df=None):
        self.df = df

    def clean(self, dropempty=False, numimpset=ImputerSettings(),
              stringimpset=ImputerSettings(basicvars="string"),
              boolimpset=ImputerSettings(basicvars="bool")):
        if self.df is not None:
            if not dropempty:
                if self.df.isnull().sum() != 0:
                    numimputer = SimpleImputer(fill_value=numimpset.fill_value, strategy=numimpset.strategy,
                                               copy=numimpset.copy, missing_values=numimpset.missing_values)
                    stringimputer = SimpleImputer(fill_value=stringimpset.fill_value, strategy=stringimpset.strategy,
                                                  copy=stringimpset.copy, missing_values=stringimpset.missing_values)
                    boolimputer = SimpleImputer(fill_value=boolimpset.fill_value, strategy=boolimpset.strategy,
                                                copy=boolimpset.copy, missing_values=boolimpset.missing_values)
                    for column in self.df:
                        if 'int' or 'float' in str(self.df[column].dtype):
                            df = numimputer.fit_transform(df[column].values.reshape(-1, 1))
                        if 'string' in str(self.df[column].dtype):
                            df = stringimputer.fit_transform(df[column].values.reshape(-1, 1))
                        if 'bool' in str(self.df[column].dtype):
                            df = boolimputer.fit_transform(df[column].values.reshape(-1, 1))
                else:
                    print("All values are empty, check the accuracy of your data!")
            else:
                self.df.dropna()
        else:
            print("Enter your data or check its accuracy !")
        return self.df

    def catencoder(self, columns, method="LabelEncoder"):
        if self.df is not None:
            for cl in columns:
                if method == "LabelEncoder":
                    encoder.fit(self.df[cl])
                    self.df[cl] = encoder.transform(self.df[cl])
                elif method == "OneHotEncoder_scikit":
                    encoder = OneHotEncoder()
                    x = encoder.fit_transform(self.df[cl].values.reshape(-1, 1)).toarray()
                    x = pd.DataFrame(x, columns=[cl + str(encoder.categories_[0][i])
                                                 for i in range(len(encoder.categories_[0]))])
                    self.df = pd.concat([self.df, x])
                    del self.df[cl]
                elif method == "OneHotEncoder_pandas":
                    self.df = pd.get_dummies(self.df, prefix=[cl], columns=[cl])
                else:
                    print("Encoder type not found!")
        else:
            print("Enter your data or check its accuracy !")
        return self.df

    def set_task(self, y):
        for col in y:
            unique = y[col].unique()
            # TODO make check smarter and check it)
            if unique[0] == 0 and unique[1] == 1:
                return [(DecisionTreeClassifier(), {'max_depth': (1, 30)})]
            else:
                return [(Ridge(), {'alpha': (1.0, 10.0)})]
