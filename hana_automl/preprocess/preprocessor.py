from hana_ml.algorithms.pal.preprocessing import (
    Imputer,
    FeatureNormalizer,
    variance_test,
)

from hana_automl.algorithms.classification.decisiontreecls import DecisionTreeCls
from hana_automl.algorithms.classification.gradboostcls import GBCls
from hana_automl.algorithms.classification.hybgradboostcls import HGBCls
from hana_automl.algorithms.classification.kneighborscls import KNeighborsCls
from hana_automl.algorithms.classification.logregressioncls import LogRegressionCls
from hana_automl.algorithms.classification.mlpcl import MLPcls
from hana_automl.algorithms.classification.naive_bayes import NBayesCls
from hana_automl.algorithms.classification.rdtclas import RDTCls
from hana_automl.algorithms.classification.svc import SVCls
from hana_automl.algorithms.regression.decisiontreereg import DecisionTreeReg
from hana_automl.algorithms.regression.glmreg import GLMReg
from hana_automl.algorithms.regression.gradboostreg import GBReg
from hana_automl.algorithms.regression.hybgradboostreg import HGBReg
from hana_automl.algorithms.regression.kneighborsreg import KNeighborsReg
from hana_automl.algorithms.regression.mlpreg import MLPreg
from hana_automl.algorithms.regression.rdtreg import RDTReg
from hana_automl.algorithms.regression.svr import SVReg
from hana_automl.utils.error import PreprocessError


class Preprocessor:
    def clean(
        self,
        data,
        droplist_columns=None,
        categorical_list=None,
        predict_column_importance=False,
        dropempty=False,
        imputer_num_strategy="mean",
        cat_strategy=None,
        normalizer_strategy=None,
        normalizer_z_score_method=None,
        normalize_int=None,
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
            imputer_num_strategy=imputer_num_strategy,
            cat_strategy=cat_strategy,
            dropempty=dropempty,
            normalizer_strategy=normalizer_strategy,
            normalizer_z_score_method=normalizer_z_score_method,
            normalize_int=normalize_int,
        )

        return df

    def autoimput(
        self,
        df=None,
        target=None,
        id=None,
        imputer_num_strategy=None,
        cat_strategy=None,
        normalizer_strategy=None,
        normalizer_z_score_method=None,
        normalize_int=None,
        dropempty=False,
        categorical_list=None,
    ):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if not dropempty:
            impute = Imputer(strategy=imputer_num_strategy)
            if categorical_list is not None:
                result = impute.fit_transform(
                    df,
                    categorical_variable=categorical_list,
                    strategy_by_col=cat_strategy,
                )
            else:
                result = impute.fit_transform(df)
            """result = self.normalize(
                result,
                normalizer_strategy,
                id,
                target,
                categorical_list=categorical_list,
                norm_int=normalize_int,
                z_score_method=normalizer_z_score_method,
            )"""
            return result
        else:
            return df.dropna()

    def removecolumns(self, columns: list, df):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if columns is not None:
            df = df.drop(columns)
        return df

    def normalize(
        self,
        df,
        method,
        id,
        target,
        categorical_list=None,
        norm_int=False,
        z_score_method="mean-standard",
    ):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if method == "min-max":
            fn = FeatureNormalizer(method="min-max", new_max=1.0, new_min=0.0)
        elif method == "z-score":
            fn = FeatureNormalizer(method="z-score", z_score_method=z_score_method)
        else:
            fn = FeatureNormalizer(method="decimal")
        col_list = df.columns
        remove_list = list()
        if categorical_list is not None:
            for i in categorical_list:
                remove_list.append(i)
        dt = df.dtypes()
        for i in dt:
            if (
                i[0] == id
                or i[0] == target
                or (not norm_int and i[1] == "INT")
                or i[1] == "CHAR"
                or i[1] == "VARCHAR"
            ):
                if not i[0] in remove_list:
                    remove_list.append(i[0])
        if len(remove_list) > 0:
            for i in remove_list:
                col_list.remove(i)
            trn = fn.fit_transform(df, key=id, features=col_list)
            new_df = df.select(*tuple(remove_list))
            df = new_df.join(
                trn.rename_columns(["ID_TEMPR", *col_list]), "ID_TEMPR=" + id
            ).deselect("ID_TEMPR")
        return df

    def autoremovecolumns(self, df):
        for cl in df:
            if (
                "object" == str(df[cl].dtype)
                and df[cl].nunique() > df[cl].shape[0] / 100 * 7
            ) or (df[cl].nunique() > df[cl].shape[0] / 100 * 9):
                df = df.drop([cl], axis=1)
        return df

    def drop_outers(self, df, id, target, cat_list):

        col_list = df.columns
        col_list.remove(id)
        col_list.remove(target)
        for i in cat_list:
            col_list.remove(i)
        data_types = df.dtypes()
        for i in data_types:
            if i[0] in col_list:
                if i[1] == "CHAR" or i[1] == "VARCHAR":
                    col_list.remove(i[0])
        for i in col_list:
            df = (
                variance_test(data=df, sigma_num=3.0, key=id, data_col=i)[0]
                .rename_columns(["ID_TEMP", "DROP"])
                .join(df, "ID_TEMP=" + id)
                .deselect("ID_TEMP")
                .filter("DROP = 0")
                .deselect("DROP")
            )
        return df

    def set_task(self, data, target, task: str, algo_exceptions=None):
        if algo_exceptions is None:
            algo_exceptions = []
        if task is None:
            if data.train.distinct(target).count() < 10:
                task = "cls"
            else:
                task = "reg"
        if task == "cls":
            if data.binomial:
                vals = data.train.select(data.target).collect()[data.target].unique()
                log = LogRegressionCls(
                    binominal=data.binomial, class_map0=vals[0], class_map1=vals[1]
                )
            else:
                log = LogRegressionCls(binominal=data.binomial)
            clslist = [
                DecisionTreeCls(),
                KNeighborsCls(),
                # log,
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
                # "LogisticRegressionClassifier": log,
                "NaiveBayesClassifier": NBayesCls(),
                "MLPClassifier": MLPcls(),
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
                # GLMReg(),
                KNeighborsReg(),
                MLPreg(),
                SVReg(),
                RDTReg(),
                GBReg(),
                HGBReg(),
            ]
            regdict = {
                "DecisionTreeRegressor": DecisionTreeReg(),
                # "GLMRegressor": GLMReg(),
                "MLPRegressor": MLPreg(),
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

    @staticmethod
    def check_binomial(df, target):
        if target is None or df is None:
            raise PreprocessError("Enter correct data for check!")
        if df.distinct(target).count() < 3:
            return True
        else:
            return False
