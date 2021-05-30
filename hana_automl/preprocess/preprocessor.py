import copy

from hana_ml import DataFrame
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
    def autoimput(
        self,
        df=None,
        target=None,
        id=None,
        imputer_num_strategy=None,
        strategy_by_col=None,
        normalizer_strategy=None,
        normalizer_z_score_method=None,
        normalize_int=None,
        categorical_list=None,
    ):
        if df is None:
            raise PreprocessError("Enter not null data!")
        impute = Imputer(strategy=imputer_num_strategy)
        if categorical_list is not None:
            if target is None:
                cols = df.columns
                cols.remove(id)
                drop = None
                for i in categorical_list:
                    if i not in cols:
                        drop = i
                if drop is not None:
                    categorical_list.remove(drop)
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
        )
        return result

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
        if norm_int:
            int_lst = []
            for i in dt:
                if target is None:
                    targ_variant = False
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
        if len(remove_list) > 0:
            for i in remove_list:
                col_list.remove(i)
        if len(col_list) > 0:
            trn: DataFrame = fn.fit_transform(df, key=id, features=col_list)
            cols: list = df.columns
            norm_cols = trn.columns
            norm_cols.remove(id)
            i = 0
            while i <= len(norm_cols) - 1:
                indx = cols.index(norm_cols[i])
                right = cols[:indx]
                left = [id]
                cur = cols[indx]
                left.extend(cols[indx + 1 :])
                left_normd = [id]
                rng = copy.copy(i)
                for o in range(rng + 1, len(norm_cols)):
                    if cols.index(norm_cols[o]) - indx == 1:
                        i += 1
                        left_normd.append(norm_cols[o])
                        left = [id]
                        indx = cols.index(norm_cols[i])
                        left.extend(cols[indx + 1 :])
                    else:
                        break
                temp_joined = (
                    df.select(*tuple(right))
                    .join(
                        trn.select(*tuple([id, cur])).rename_columns(["ID_TEMPR", cur]),
                        "ID_TEMPR=" + id,
                    )
                    .deselect("ID_TEMPR")
                )
                if len(left_normd) > 1:
                    temp_joined = temp_joined.join(
                        trn.select(*tuple(left_normd)).rename_columns(
                            ["ID_TEMPR", *left_normd[1:]]
                        ),
                        "ID_TEMPR=" + id,
                    ).deselect("ID_TEMPR")
                df = temp_joined.join(
                    df.select(*tuple(left)).rename_columns(["ID_TEMPR", *left[1:]]),
                    "ID_TEMPR=" + id,
                ).deselect("ID_TEMPR")
                i = i + 1
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
        if cat_list is not None:
            for i in cat_list:
                if i in col_list:
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
                "KNeighborsRegressor": KNeighborsReg(),
                "DecisionTreeRegressor": DecisionTreeReg(),
                # "GLMRegressor": GLMReg(),
                "MLPRegressor": MLPreg(),
                "RDTRegressor": RDTReg(),
                "SupportVectorRegressor": SVReg(),
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
