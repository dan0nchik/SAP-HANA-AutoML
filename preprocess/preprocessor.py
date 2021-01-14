import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from algorithms.classification import DecisionTree
from preprocess.impsettings import ImputerSettings


class Preprocessor:

    def clean(self, data, droplist_columns=None, categorlist=None, predict_column_importance=True, dropempty=False,
              encoder_method="OneHotEncoder_pandas",
              numimpset=ImputerSettings(),
              stringimpset=ImputerSettings(basicvars="string"),
              boolimpset=ImputerSettings(basicvars="bool")):
        if data is None:
            print('Enter not null data!')
            return data
        if droplist_columns is not None or predict_column_importance:
            data.X_train = self.removecolumns(columns=droplist_columns, autoremove=predict_column_importance,
                                              df=data.X_train)
            data.X_test = self.removecolumns(columns=droplist_columns, autoremove=predict_column_importance,
                                             df=data.X_test)
        data.X_train = self.autoimput(df=data.X_train, dropempty=dropempty, numimpset=numimpset,
                                      stringimpset=stringimpset,
                                      boolimpset=boolimpset)
        data.X_test = self.autoimput(df=data.X_test, dropempty=dropempty, numimpset=numimpset,
                                     stringimpset=stringimpset,
                                     boolimpset=boolimpset)
        data.X_train = self.catencoder(df=data.X_train, columns=categorlist, method=encoder_method)
        data.X_test = self.catencoder(df=data.X_test, columns=categorlist, method=encoder_method)

        return data

    def autoimput(self, df, numimpset, stringimpset, boolimpset, dropempty=False):
        if df is None:
            print('Enter not null data!')
            return df
        if not dropempty:
            numimputer = SimpleImputer(fill_value=numimpset.fill_value, strategy=numimpset.strategy,
                                       missing_values=numimpset.missing_values)
            stringimputer = SimpleImputer(fill_value=stringimpset.fill_value, strategy=stringimpset.strategy,
                                          missing_values=stringimpset.missing_values)
            boolimputer = SimpleImputer(fill_value=boolimpset.fill_value, strategy=boolimpset.strategy,
                                        missing_values=boolimpset.missing_values)
            for column in df:
                df2 = df.copy()
                dtype = str(df[column].dtype)
                if 'object' == dtype:
                    df2[column] = stringimputer.fit_transform(df[column].values.reshape(-1, 1))[:,0]
                    df = df2.copy()
                if 'uint8' == dtype or 'float64' == dtype or 'int64' == dtype:
                    df2[column] = numimputer.fit_transform(df[column].values.reshape(-1, 1))[:,0]
                    df = df2.copy()
                if 'bool' == dtype:
                    df2[column] = boolimputer.fit_transform(df[column].values.reshape(-1, 1))[:,0]
                    df = df2.copy()
        else:
            df.dropna()
        return df

    def catencoder(self, columns, df, method):
        if df is None:
            print('Enter not null data!')
            return df
        for cl in columns:
            if method == "LabelEncoder":
                encoder = LabelEncoder()
                encoder.fit(df[cl])
                df[cl] = encoder.transform(df[cl])
            elif method == "OneHotEncoder_scikit":
                encoder = OneHotEncoder()
                x = encoder.fit_transform(df[cl].values.reshape(-1, 1)).toarray()
                x = pd.DataFrame(x, columns=[cl + str(encoder.categories_[0][i])
                                             for i in range(len(encoder.categories_[0]))])
                df = pd.concat([df, x])
                del df[cl]
            elif method == "OneHotEncoder_pandas":
                df = pd.get_dummies(df, prefix=[cl], columns=[cl])
            else:
                print("Encoder type not found!")
        return df

    def removecolumns(self, columns, autoremove, df):
        if df is None:
            print('Enter not null data!')
            return df
        if columns is not None:
            for cl in df:
                if cl in columns:
                    df = df.drop([cl], axis=1)
        if autoremove:
            for cl in df:
                if 'object' == str(df[cl].dtype) and df[cl].nunique() > df[cl].shape[0] / 100 * 7:
                    df = df.drop([cl], axis=1)
        return df

    def set_task(self, y, algo_exceptions=None):
        if algo_exceptions is None:
            algo_exceptions = []
        for col in y:
            if y[col].nunique() == 2:
                clslist = [DecisionTree()]
                if 'DecisionTree' in algo_exceptions:
                    clslist.remove(DecisionTree())
                return clslist, 'cls'
            elif y[col].nunique() < 10:
                clslist = [DecisionTree()]
                if 'DecisionTree' in algo_exceptions:
                    clslist.remove(DecisionTree())
                return clslist, 'cls'
            else:
                reglist = [Ridge()]
                if 'Ridge' in algo_exceptions:
                    reglist.remove(Ridge())
                return reglist, 'reg'
