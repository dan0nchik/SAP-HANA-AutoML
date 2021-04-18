from hana_ml.algorithms.pal.metrics import accuracy_score
from hana_ml.dataframe import create_dataframe_from_pandas

from algorithms.ensembles.bagging import Bagging
from pipeline.leaderboard import Leaderboard
from preprocess.preprocessor import Preprocessor
import pandas as pd
from utils.error import BaggingError


class BaggingCls(Bagging):
    def __init__(self, categorical_features, id_col, connection_context, table_name, model_list: list = None,
                 leaderboard: Leaderboard = None):
        super(BaggingCls, self).__init__(categorical_features, id_col, connection_context,
                                         table_name, model_list, leaderboard)
        self.title = "BaggingClassifier"

    def score(self, data=None, df=None):
        predictions = list()
        if data is None and df is None:
            raise BaggingError("Provide valid data for accuracy estimation")
        pr = Preprocessor()
        for model in self.model_list:
            if df is not None:
                df2 = pr.clean(
                    data=df, num_strategy=model.preprocessor['imputer']
                )
            else:
                df2 = pr.clean(
                    data=data.valid.drop(data.target), num_strategy=model.preprocessor['imputer']
                )
            pred = model.algorithm.model.predict(df2, self.id_col)
            if type(pred) == tuple:
                predictions.append(pred[0])
            else:
                predictions.append(pred)
        pd_res = list()
        for res in predictions:
            pd_res.append(res.collect())
        pred = list()
        for i in range(pd_res[0].shape[0]):
            if pd_res[0].at[i, pd_res[0].columns[1]] == pd_res[1].at[i, pd_res[1].columns[1]]:
                pred.append(pd_res[0].at[i, pd_res[0].columns[1]])
            else:
                pred.append(pd_res[2].at[i, pd_res[2].columns[1]])
        d = {'PREDICTION': pred}
        df = pd.DataFrame(data=d)
        hana_df = create_dataframe_from_pandas(
            self.connection_context, df, self.table_name + "_bagging", force=True, drop_exist_tab=True
        )
        ftr: list = data.valid.columns
        ftr.remove(data.target)
        dat = data.valid.drop(ftr)
        itg = hana_df.join(dat, "1 = 1")
        print(itg.collect().head())
        return accuracy_score(data=itg, label_true=data.target, label_pred="PREDICTION")

    def get_models(self):
        return self.model_list

    def predict(self, cat):
        pass
