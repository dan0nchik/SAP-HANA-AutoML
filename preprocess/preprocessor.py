from hana_ml.algorithms.pal.preprocessing import Imputer

from algorithms.classification.decisiontreecls import DecisionTreeCls
from algorithms.classification.gradboostcls import GBCls
from algorithms.classification.hybgradboostcls import HGBCls
from algorithms.classification.kneighborscls import KNeighborsCls
from algorithms.classification.logregressioncls import LogRegressionCls
from algorithms.classification.mlpcl import MLPcls
from algorithms.classification.naive_bayes import NBayesCls
from algorithms.classification.rdtclas import RDTCls
from algorithms.classification.svc import SVCls
from algorithms.regression.decisiontreereg import DecisionTreeReg
from algorithms.regression.glmreg import GLMReg
from algorithms.regression.gradboostreg import GBReg
from algorithms.regression.hybgradboostreg import HGBReg
from algorithms.regression.kneighborsreg import KNeighborsReg
from algorithms.regression.mlpreg import MLPreg
from algorithms.regression.expreg import ExponentialReg
from algorithms.regression.rdtreg import RDTReg
from algorithms.regression.svr import SVReg
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
        cat_strategy=None,
    ):
        if cat_strategy is None:
            cat_strategy = []
        if data is None:
            raise PreprocessError("Enter not null data!")
        if predict_column_importance:
            None
            # TODO: AutoRemove
        df = self.autoimput(
            data,
            num_strategy=num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
        )

        return df

    def autoimput(
        self, df, num_strategy, cat_strategy, dropempty=False, categorical_list=None
    ):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if not dropempty:
            impute = Imputer(strategy=num_strategy)
            if categorical_list is not None:
                result = impute.fit_transform(
                    df,
                    categorical_variable=categorical_list,
                    strategy_by_col=cat_strategy,
                )
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
            clslist = [
                DecisionTreeCls(),
                LogRegressionCls(),
                NBayesCls(),
                MLPcls(),
                SVCls(),
                RDTCls(),
                GBCls(),
                HGBCls(),
            ]
            clsdict = {
                "KNeighborsClassifier": KNeighborsCls(),
                "DecisionTreeClassifier": DecisionTreeCls(),
                "LogisticRegressionClassifier": LogRegressionCls(),
                "NaiveBayesClassifier": NBayesCls(),
                # "MLPClassifier": MLPcls(),
                "SVClassifier": SVCls(),
                "RDTClassifier": RDTCls(),
                "GradientBoostingClassifier": GBCls(),
                "HybridGradientBoostingClassifier": HGBCls(),
            }
            clslist = [i for i in clslist if i.title not in algo_exceptions]
            clsdict = {
                key: value
                for key, value in clsdict.items()
                if key.title not in algo_exceptions
            }
            return clslist, "cls", clsdict
        else:
            reglist = [
                DecisionTreeReg(),
                GLMReg(),
                KNeighborsReg(),
                MLPreg(),
                SVReg(),
                RDTReg(),
                GBReg(),
                HGBReg(),
            ]
            regdict = {
                "DecisionTreeRegressor": DecisionTreeReg(),
                "GLMRegressor": GLMReg(),
                # "MLPRegressor": MLPreg(),
                "KNeighborsRegressor": KNeighborsReg(),
                "SupportVectorRegressor": SVReg(),
                "RDTRegressor": RDTReg(),
                "GradientBoostingRegressor": GBReg(),
                "HybridGradientBoostingRegressor": HGBReg(),
            }
            reglist = [i for i in reglist if i.title not in algo_exceptions]
            regdict = {
                key: value
                for key, value in regdict.items()
                if key.title not in algo_exceptions
            }
            return reglist, "reg", regdict
