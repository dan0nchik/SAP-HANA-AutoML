from hana_ml.algorithms.pal.metrics import accuracy_score
from hana_ml.dataframe import create_dataframe_from_pandas

from hana_automl.algorithms.ensembles.blending import Blending
from hana_automl.pipeline.leaderboard import Leaderboard


class BlendingCls(Blending):
    def __init__(
        self,
        categorical_features,
        id_col,
        connection_context,
        table_name,
        model_list: list = None,
        leaderboard: Leaderboard = None,
    ):
        super(BlendingCls, self).__init__(
            categorical_features,
            id_col,
            connection_context,
            table_name,
            model_list,
            leaderboard,
        )
        self.title = "BlendingClassifier"

    def score(self, data):
        return self.inner_score(data, key=data.id_colm, label=data.target)

    def inner_score(self, data, key, label=None):
        cols = data.valid.columns
        cols.remove(key)
        if label is not None:
            cols.remove(label)
        prediction = self.predict(data=data)
        prediction = prediction.select("ID", "PREDICTION").rename_columns(
            ["ID_P", "PREDICTION"]
        )
        actual = data.valid.select(key, label).rename_columns(["ID_A", "ACTUAL"])
        joined = actual.join(prediction, "ID_P=ID_A").select("ACTUAL", "PREDICTION")
        return accuracy_score(joined, label_true="ACTUAL", label_pred="PREDICTION")

    def predict(self, data=None, df=None, id_colm=None):
        predictions = super(BlendingCls, self).predict(data=data, df=df)
        pd_res = list()
        if id_colm is None:
            id_colm = data.id_colm
        for i in range(len(predictions)):
            k = (
                predictions[i]
                .select(id_colm, predictions[i].columns[1])
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
        hana_df = create_dataframe_from_pandas(
            self.connection_context,
            joined,
            self.table_name + "_sclsblending",
            force=True,
            drop_exist_tab=True,
            disable_progressbar=True,
        )
        return hana_df
