import hana_ml
from hana_ml.algorithms.pal.metrics import r2_score

from hana_automl.algorithms.ensembles.blending import Blending
from hana_automl.metric.mae import mae_score
from hana_automl.metric.mse import mse_score
from hana_automl.metric.rmse import rmse_score
from hana_automl.pipeline.data import Data


class BlendingReg(Blending):
    def __init__(
        self,
        id_col: str = None,
        connection_context: hana_ml.dataframe.ConnectionContext = None,
        table_name: str = None,
        model_list: list = None,
        leaderboard: list = None,
    ):
        super().__init__(
            id_col,
            connection_context,
            table_name,
            model_list,
            leaderboard,
        )
        self.title = "BlendingRegressor"

    def predict(
        self, data: Data = None, df: hana_ml.DataFrame = None, id_colm: str = None
    ):
        if id_colm is None:
            id_colm = data.id_colm
        predictions = super(BlendingReg, self).predict(data=data, df=df)
        pd_res = list()
        for i in range(len(predictions)):
            if (
                str(self.model_list[i].algorithm.model).split(" ")[0]
                == "<hana_ml.algorithms.pal.neural_network.MLPRegressor"
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
        joined = joined.select(
            "ID", ("(PREDICTION1 + PREDICTION2 + PREDICTION3)/3", "PREDICTION")
        )
        return joined

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
        if metric == "r2_score":
            return r2_score(joined, label_true="ACTUAL", label_pred="PREDICTION")
        if metric == "mae":
            return mae_score(df=joined)
        if metric == "mse":
            return mse_score(df=joined)
        if metric == "rmse":
            return rmse_score(df=joined)
