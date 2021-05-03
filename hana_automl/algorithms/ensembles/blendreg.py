import pandas as pd
from hana_ml.algorithms.pal import metrics
from hana_ml.algorithms.pal.metrics import r2_score
from hana_ml.dataframe import create_dataframe_from_pandas

from hana_automl.algorithms.ensembles.blending import Blending
from hana_automl.pipeline.leaderboard import Leaderboard


class BlendingReg(Blending):
    def __init__(
        self,
        categorical_features,
        id_col,
        connection_context,
        table_name,
        model_list: list = None,
        leaderboard: Leaderboard = None,
    ):
        super(BlendingReg, self).__init__(
            categorical_features,
            id_col,
            connection_context,
            table_name,
            model_list,
            leaderboard,
        )
        self.title = "BlendingRegressor"

    def predict(self, data=None, df=None, id_colm=None):
        if id_colm is None:
            id_colm = data.id_colm
        predictions = super(BlendingReg, self).predict(data=data, df=df)
        pd_res = list()
        for i in range(len(predictions)):
            k = predictions[i].select(id_colm,
                                      predictions[i].columns[1]).rename_columns(['ID_'+str(i), 'PREDICTION'+str(i)])
            pd_res.append(k)
        joined = pd_res[0].join(pd_res[1], 'ID_0=ID_1').select('ID_0', 'PREDICTION0', 'PREDICTION1').join(
            pd_res[2], 'ID_0=ID_2').select('ID_0', 'PREDICTION0', 'PREDICTION1', 'PREDICTION2')
        joined = joined.rename_columns(['ID', 'PREDICTION1', 'PREDICTION2', 'PREDICTION3'])
        joined = joined.select('ID', ('(PREDICTION1 + PREDICTION2 + PREDICTION3)/3', 'PREDICTION'))
        return joined

    def score(self, data):
        return self.inner_score(data, key=data.id_colm, label=data.target)

    def inner_score(self, data, key, label=None):
        cols = data.valid.columns
        cols.remove(key)
        cols.remove(label)
        prediction = self.predict(data=data)
        prediction = prediction.select('ID', 'PREDICTION').rename_columns(['ID_P', 'PREDICTION'])
        actual = data.valid.select(key, label).rename_columns(['ID_A', 'ACTUAL'])
        joined = actual.join(prediction, 'ID_P=ID_A').select('ACTUAL', 'PREDICTION')
        return metrics.r2_score(joined,
                                      label_true='ACTUAL',
                                      label_pred='PREDICTION')

