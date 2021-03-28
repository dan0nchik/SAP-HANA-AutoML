import pandas as pd
from hana_ml.algorithms.pal.preprocessing import Imputer

from algorithms.classification.decisiontreecls import DecisionTreeCls
from algorithms.classification.kneighbors import KNeighbors
from algorithms.classification.logregression import LogRegression
from algorithms.regression.decisiontreereg import DecisionTreeReg
from algorithms.regression.glmreg import GLMRegression
from algorithms.regression.kneighborsreg import KNeighborsReg
from preprocess.impsettings import ImputerSettings
from utils.error import PreprocessError


class Preprocessor:
    def clean(
            self,
            data=None,
            droplist_columns=None,
            categorical_list=None,
            predict_column_importance=False,
            dropempty=False,
            num_strategy="mean",
            cat_strategy=[]
    ):
        if data is None:
            raise PreprocessError("Enter not null data!")
        if predict_column_importance:
            None
            # TODO: AutoRemove
        self.autoimput(data, num_strategy=num_strategy, cat_strategy=cat_strategy, dropempty=dropempty)

        return data

    def autoimput(self, df, num_strategy, cat_strategy, dropempty=False, categorical_list=None):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if not dropempty:
            impute = Imputer(strategy=num_strategy)
            if categorical_list is not None:
                result = impute.fit_transform(df, categorical_variable=categorical_list,
                                              strategy_by_col=cat_strategy)
            else:
                result = impute.fit_transform(df)
            return result
        else:
            return df.dropna()

    def removecolumns(self, columns: list, df):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if columns is not None:
            df = df.drop(columns)
        return df

    def autoremovecolumns(self, df):
        for cl in df:
            if (
                    "object" == str(df[cl].dtype)
                    and df[cl].nunique() > df[cl].shape[0] / 100 * 7
            ) or (df[cl].nunique() > df[cl].shape[0] / 100 * 9):
                df = df.drop([cl], axis=1)
        return df

    def set_task(self, data, target, algo_exceptions=None):
        if algo_exceptions is None:
            algo_exceptions = []
        if data.train.distinct(target).count() < 10:
            clslist = [DecisionTreeCls(), LogRegression()]
            clsdict = {"DecisionTree": DecisionTreeCls(), "Logistic Regression": LogRegression()}
            if "DecisionTree" in algo_exceptions:
                clslist.remove(DecisionTreeCls())
                clsdict.pop("DecisionTree")
            if "Logistic Regression" in algo_exceptions:
                clslist.remove(LogRegression())
                clsdict.pop("Logistic Regression")
            if "KNeighbors" in algo_exceptions:
                clslist.remove(KNeighbors())
                clsdict.pop("KNeighbors")
            return clslist, 'cls', clsdict
        else:
            reglist = [DecisionTreeReg(), GLMRegression()]
            regdict = {"DecisionTreeReg": DecisionTreeReg(), "GLMRegression": GLMRegression()}
            if "DecisionTreeReg" in algo_exceptions:
                reglist.remove(DecisionTreeReg())
                regdict.pop("DecisionTreeReg")
            if "GLMRegression" in algo_exceptions:
                reglist.remove(GLMRegression())
                regdict.pop("GLMRegression")
            if "KNNRegressor" in algo_exceptions:
                reglist.remove(KNeighborsReg())
                regdict.pop("KNRegressor")
            return reglist, 'reg', regdict
