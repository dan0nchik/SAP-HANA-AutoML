from hana_automl.pipeline.leaderboard import Leaderboard
from hana_automl.preprocess.preprocessor import Preprocessor
from hana_automl.utils.error import BlendingError


class Blending:
    def __init__(
        self,
        categorical_features,
        id_col,
        connection_context,
        table_name,
        model_list: list = None,
        leaderboard: Leaderboard = None,
    ):
        self.id_col = id_col
        self.categorical_features = categorical_features
        self.title = ""
        self.name = "Untitled"  # for storage
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
            self.model_list = leaderboard.board[:3]

    def score(self, data):
        pass

    def predict(self, data, df):
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
                    imputer_num_strategy=model.preprocessor.tuned_num_strategy,
                    cat_strategy=None,
                    dropempty=None,
                    categorical_list=self.categorical_features,
                    normalizer_strategy=model.preprocessor.tuned_normalizer_strategy,
                    normalizer_z_score_method=model.preprocessor.tuned_z_score_method,
                    normalize_int=model.preprocessor.tuned_normalize_int,
                )
            else:
                df2 = pr.autoimput(
                    df=data.valid.drop(data.target),
                    id=data.id_colm,
                    target=data.target,
                    imputer_num_strategy=model.preprocessor.tuned_num_strategy,
                    cat_strategy=None,
                    dropempty=None,
                    categorical_list=self.categorical_features,
                    normalizer_strategy=model.preprocessor.tuned_normalizer_strategy,
                    normalizer_z_score_method=model.preprocessor.tuned_z_score_method,
                    normalize_int=model.preprocessor.tuned_normalize_int,
                )
            pred = model.algorithm.model.predict(df2, self.id_col)
            if type(pred) == tuple:
                predictions.append(pred[0])
            else:
                predictions.append(pred)
        return predictions
