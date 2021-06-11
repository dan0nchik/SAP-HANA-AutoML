import copy
import math
from hana_ml import DataFrame
from hana_ml.algorithms.pal.neighbors import KNNRegressor
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
from hana_automl.algorithms.regression.expreg import ExponentialReg
from hana_automl.algorithms.regression.glmreg import GLMReg
from hana_automl.algorithms.regression.gradboostreg import GBReg
from hana_automl.algorithms.regression.hybgradboostreg import HGBReg
from hana_automl.algorithms.regression.kneighborsreg import KNeighborsReg
from hana_automl.algorithms.regression.mlpreg import MLPreg
from hana_automl.algorithms.regression.rdtreg import RDTReg
from hana_automl.algorithms.regression.svr import SVReg
from hana_automl.utils.error import PreprocessError


class Preprocessor:
    def __init__(self):
        self.clsdict = {
            "KNeighborsClassifier": KNeighborsCls(),
            "DecisionTreeClassifier": DecisionTreeCls(),
            # "LogisticRegressionClassifier": log,
            "NaiveBayesClassifier": NBayesCls(),
            "MLPClassifier": MLPcls(),
            "SupportVectorClassifier": SVCls(),
            "RandomDecisionTreeClassifier": RDTCls(),
            "GradientBoostingClassifier": GBCls(),
            "HybridGradientBoostingClassifier": HGBCls(),
        }
        self.regdict = {
            "DecisionTreeRegressor": DecisionTreeReg(),
            # "GLMRegressor": GLMReg(),
            "ExponentialRegressor": ExponentialReg(),
            "MLPRegressor": MLPreg(),
            "Random_Decision_Tree_Regressor": RDTReg(),
            "SupportVectorRegressor": SVReg(),
            "GradientBoostingRegressor": GBReg(),
            "HybridGradientBoostingRegressor": HGBReg(),
            "KNNRegressor": KNeighborsReg(),
        }

    def autoimput(
        self,
        df: DataFrame = None,
        target: str = None,
        id: str = None,
        imputer_num_strategy: str = None,
        strategy_by_col: str = None,
        normalizer_strategy: str = None,
        normalizer_z_score_method: str = None,
        normalize_int: bool = None,
        categorical_list: list = None,
        normalization_excp: list = None,
    ):
        if df is None:
            raise PreprocessError("Enter not null data!")
        impute = Imputer(strategy=imputer_num_strategy)
        if categorical_list is not None:
            categorical_list = list(set(categorical_list))
            if target is None:
                cols = df.columns
                for column in categorical_list:
                    if column not in cols:
                        categorical_list.remove(column)
            if strategy_by_col is not None:
                result = impute.fit_transform(
                    df,
                    categorical_variable=categorical_list,
                    strategy_by_col=strategy_by_col,
                )
            else:
                result = impute.fit_transform(
                    df,
                    categorical_variable=categorical_list,
                )
        else:
            if strategy_by_col is not None:
                result = impute.fit_transform(df, strategy_by_col=strategy_by_col)
            else:
                result = impute.fit_transform(df)
        result = self.normalize(
            result,
            normalizer_strategy,
            id,
            target,
            categorical_list=categorical_list,
            norm_int=normalize_int,
            z_score_method=normalizer_z_score_method,
            normalization_excp=normalization_excp,
        )
        return result

    def removecolumns(self, columns: list, df: DataFrame):
        if df is None:
            raise PreprocessError("Enter not null data!")
        if columns is not None:
            df = df.drop(columns)
        return df

    def normalize(
        self,
        df: DataFrame,
        method: str,
        id: str,
        target: str,
        categorical_list: list = None,
        norm_int: bool = False,
        z_score_method: str = "mean-standard",
        normalization_excp=None,
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
        if categorical_list is not None and len(categorical_list) > 0:
            for i in categorical_list:
                remove_list.append(i)
        else:
            categorical_list = []
        dt = df.dtypes()
        if norm_int:
            int_lst = []
            for i in dt:
                if target is None:
                    targ_variant = True
                else:
                    targ_variant = i[0] != target
                if (
                    i[0] != id
                    and (i[1] in ["INT", "SMALLINT", "MEDIUMINT", "INTEGER", "BIGINT"])
                    and targ_variant
                    and not (i[0] in categorical_list)
                ):
                    int_lst.append(i[0])
            if len(int_lst) > 0:
                df = df.cast(int_lst, "DOUBLE")
            dt = df.dtypes()
        for i in dt:
            if target is None:
                targ_variant = False
            else:
                targ_variant = i[0] == target
            if i[0] == id or targ_variant or i[1] in ["INT", "CHAR", "VARCHAR"]:
                if not i[0] in remove_list:
                    remove_list.append(i[0])
        if normalization_excp is not None:
            for i in normalization_excp:
                if i not in remove_list:
                    remove_list.append(i)
        if len(remove_list) > 0:
            for i in remove_list:
                col_list.remove(i)
        if len(col_list) > 0:
            trn: DataFrame = fn.fit_transform(df, key=id, features=col_list)
            trn = trn.rename_columns({id: "TEMP_ID"})
            df = df.drop(col_list)
            df = (
                df.alias("SOURCE")
                .join(
                    trn.alias("NORMALIZED"),
                    f"NORMALIZED.TEMP_ID = SOURCE.{id}",
                    how="inner",
                )
                .drop(["TEMP_ID"])
            )
        return df

    def autoremovecolumns(self, df: DataFrame):
        for column in df.columns:
            if (
                "object" == str(df[column].dtype)
                and df[column].nunique() > df[column].shape[0] / 100 * 7
            ) or (df[column].nunique() > df[column].shape[0] / 100 * 9):
                df = df.drop([column])
        return df

    def drop_outers(self, df: DataFrame, id: str, target: str, cat_list: list):
        col_list = df.columns
        if cat_list is None:
            cat_list = []
        cat_list.append(target)
        cat_list.append(id)
        col_list = list(filter(lambda column: column not in cat_list, col_list))
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

    def set_task(self, data, target: str, task: str, algo_exceptions=None):
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
            self.clslist = [
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
            clslist = [i for i in self.clslist if i.title not in algo_exceptions]
            clsdict = {
                key: value
                for key, value in self.clsdict.items()
                if key.title not in algo_exceptions
            }
            return clslist, "cls", clsdict
        else:
            self.reglist = [
                DecisionTreeReg(),
                # GLMReg(),
                KNeighborsReg(),
                MLPreg(),
                SVReg(),
                RDTReg(),
                GBReg(),
                HGBReg(),
            ]
            reglist = [i for i in self.reglist if i.title not in algo_exceptions]
            regdict = {
                key: value
                for key, value in self.regdict.items()
                if key.title not in algo_exceptions
            }
            return reglist, "reg", regdict

    @staticmethod
    def check_binomial(df: DataFrame, target: str):
        if target is None or df is None:
            raise PreprocessError("Enter correct data for check!")
        if df.distinct(target).count() < 3:
            return True
        else:
            return False

    @staticmethod
    def check_normalization_exceptions(df, id, target, categorical_list):
        excpt_list = []
        dts = df.columns
        dts.remove(id)
        dts.remove(target)
        if categorical_list is None:
            categorical_list = []
        for dt in dts:
            if (
                df.is_numeric(dt)
                and dt not in categorical_list
                and df.distinct(dt).count() < 3
            ):
                excpt_list.append(dt)
        if len(excpt_list) < 1:
            return None
        else:
            return excpt_list
