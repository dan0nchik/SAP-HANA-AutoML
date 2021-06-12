import hana_ml
from hana_ml.algorithms.pal.metrics import accuracy_score
from hana_ml.dataframe import create_dataframe_from_pandas

from hana_automl.algorithms.ensembles.blending import Blending
from hana_automl.pipeline.data import Data


class BlendingCls(Blending):
    def __init__(
        self,
        id_col: str = None,
        connection_context: hana_ml.ConnectionContext = None,
        table_name: str = None,
        model_list: list = None,
        leaderboard: list = None,
    ):
        super(BlendingCls, self).__init__(
            id_col,
            connection_context,
            table_name,
            model_list,
            leaderboard,
        )
        self.title: str = "BlendingClassifier"

    def score(self, data: Data, metric: str):
        return self.inner_score(
            data, key=data.id_colm, metric=metric, label=data.target
        )

    def inner_score(self, data: Data, key: str, metric: str, label: str = None):
        prediction = self.predict(data=data)
        prediction = prediction.select("ID", "PREDICTION").rename_columns(
            ["ID_P", "PREDICTION"]
        )
        actual = data.valid.select(key, label).rename_columns(["ID_A", "ACTUAL"])
        joined = actual.join(prediction, "ID_P=ID_A").select("ACTUAL", "PREDICTION")
        if metric == "accuracy":
            return accuracy_score(joined, label_true="ACTUAL", label_pred="PREDICTION")

    def predict(
        self, data: Data = None, df: hana_ml.DataFrame = None, id_colm: str = None
    ):
        predictions = super(BlendingCls, self).predict(data=data, df=df)
        pd_res = list()
        if id_colm is None:
            id_colm = data.id_colm
        for i in range(len(predictions)):
            if (
                str(self.model_list[i].algorithm.model).split(" ")[0]
                == "<hana_ml.algorithms.pal.neural_network.MLPClassifier"
            ):
                id_val = 2
            else:
                id_val = 1
            k = (
                predictions[i]
                .select(id_colm, predictions[i].columns[id_val])
                .rename_columns(["ID_" + str(i), "PREDICTION" + str(i)])
            )
            pd_res.append(k)
        joined = (
            pd_res[0]
            .join(pd_res[1], "ID_0=ID_1")
            .select("ID_0", "PREDICTION0", "PREDICTION1")
            .join(pd_res[2], "ID_0=ID_2")
            .select("ID_0", "PREDICTION0", "PREDICTION1", "PREDICTION2")
        )
        joined = joined.rename_columns(
            ["ID", "PREDICTION1", "PREDICTION2", "PREDICTION3"]
        )
        joined = joined.collect()
        k = list()
        for i in range(joined.shape[0]):
            if joined.at[i, "PREDICTION1"] == joined.at[i, "PREDICTION2"]:
                k.append(joined.at[i, "PREDICTION1"])
            else:
                k.append(joined.at[i, "PREDICTION3"])
        joined.insert(4, "PREDICTION", k, True)
        if self.table_name is None:
            self.table_name = "#TEMP_TABLE"
        hana_df = create_dataframe_from_pandas(
            self.connection_context,
            joined,
            self.table_name + "_sclsblending",
            force=True,
            drop_exist_tab=True,
            disable_progressbar=True,
        )
        return hana_df.deselect(["PREDICTION1", "PREDICTION2", "PREDICTION3"])
