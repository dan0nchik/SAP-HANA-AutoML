import hana_ml

from hana_automl.pipeline.data import Data
from hana_automl.preprocess.preprocessor import Preprocessor
from hana_automl.utils.error import BlendingError


class Blending:
    def __init__(
        self,
        id_col: str = None,
        connection_context: hana_ml.ConnectionContext = None,
        table_name: str = None,
        model_list: list = None,
        leaderboard: list = None,
    ):
        self.id_col = id_col
        self.title = ""
        self.name = None  # for storage
        self.version = 1  # for storage
        self.table_name = table_name
        self.connection_context = connection_context
        if model_list is None and leaderboard is None:
            raise BlendingError(
                "Provide list of models or a leaderboard for ensemble creation"
            )
        if model_list is not None:
            self.model_list = model_list
        else:
            self.model_list = leaderboard[:3]

    def score(self, data: Data, metric: str):
        pass

    def predict(self, data: Data, df: hana_ml.DataFrame):
        predictions = list()
        if data is None and df is None:
            raise BlendingError("Provide valid data for accuracy estimation")
        pr = Preprocessor()
        for model in self.model_list:
            if df is not None:
                df2 = pr.autoimput(
                    df=df,
                    id=self.id_col,
                    target=data.target,
                    strategy_by_col=model.preprocessor.strategy_by_col,
                    imputer_num_strategy=model.preprocessor.tuned_num_strategy,
                    normalizer_strategy=model.preprocessor.tuned_normalizer_strategy,
                    normalizer_z_score_method=model.preprocessor.tuned_z_score_method,
                    normalize_int=model.preprocessor.tuned_normalize_int,
                    categorical_list=model.preprocessor.categorical_cols,
                    normalization_excp=model.preprocessor.normalization_exceptions,
                )
            else:
                if data.target is None:
                    dt = data.valid
                else:
                    dt = data.valid.drop([data.target])
                df2 = pr.autoimput(
                    df=dt,
                    id=data.id_colm,
                    target=None,
                    strategy_by_col=model.preprocessor.strategy_by_col,
                    imputer_num_strategy=model.preprocessor.tuned_num_strategy,
                    normalizer_strategy=model.preprocessor.tuned_normalizer_strategy,
                    normalizer_z_score_method=model.preprocessor.tuned_z_score_method,
                    normalize_int=model.preprocessor.tuned_normalize_int,
                    categorical_list=model.preprocessor.categorical_cols,
                    normalization_excp=model.preprocessor.normalization_exceptions,
                )

            pred = model.algorithm.model.predict(df2, self.id_col)
            if type(pred) == tuple:
                predictions.append(pred[0])
            else:
                predictions.append(pred)
        return predictions
