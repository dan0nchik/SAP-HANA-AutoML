import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from preprocess.impsettings import ImputerSettings


class Preprocessor:
    def __init__(self, df=None):
        self.df = df

    def clean(self, df=None, dropempty=False, numimpset=ImputerSettings(),
              stringimpset=ImputerSettings(basicvars="string"),
              boolimpset=ImputerSettings(basicvars="bool")):
        if not (df is None):
            self.df = df
        if self.df is not None:
            if not dropempty:
                numimputer = SimpleImputer(fill_value=numimpset.fill_value, strategy=numimpset.strategy,
                                           copy=numimpset.copy, missing_values=numimpset.missing_values)
                stringimputer = SimpleImputer(fill_value=stringimpset.fill_value, strategy=stringimpset.strategy,
                                              copy=stringimpset.copy, missing_values=stringimpset.missing_values)
                boolimputer = SimpleImputer(fill_value=boolimpset.fill_value, strategy=boolimpset.strategy,
                                            copy=boolimpset.copy, missing_values=boolimpset.missing_values)
                for column in self.df:
                    df2 = df.copy()
                    if 'int' or 'float' in str(self.df[column].dtype):
                        df2[[column]] = numimputer.fit_transform(df[[column]].values.reshape(-1, 1))
                        df = df2.copy()
                    if 'string' in str(self.df[column].dtype):
                        df2[[column]] = stringimputer.fit_transform(df[[column]].values.reshape(-1, 1))
                        df = df2.copy()
                    if 'bool' in str(self.df[column].dtype):
                        df2[[column]] = boolimputer.fit_transform(df[[column]].values.reshape(-1, 1))
                        df = df2.copy()
            else:
                self.df.dropna()
        else:
            print("Enter your data or check its accuracy !")
        return self.df

    def catencoder(self, columns, df=None, method="LabelEncoder"):
        if not (df is None):
            self.df = df
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
            if unique[0] == 0 and unique[1] == 1:
                return 'cls'
            elif y[col].nunique() < 10:
                return 'cls'
                # TODO: Not Binary classification
            else:
                return 'reg'
