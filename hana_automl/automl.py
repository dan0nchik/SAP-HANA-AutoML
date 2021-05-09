import pandas as pd
import json

from hana_automl.algorithms.ensembles.blendcls import BlendingCls
from hana_automl.algorithms.ensembles.blendreg import BlendingReg
from hana_automl.pipeline.input import Input
from hana_automl.pipeline.pipeline import Pipeline
from hana_automl.preprocess.preprocessor import Preprocessor
from hana_automl.preprocess.settings import PreprocessorSettings
from hana_automl.utils.error import AutoMLError, BlendingError
from hana_ml.model_storage import ModelStorage
import hana_ml


class AutoML:
    """Main class. Control the whole Automated Machine Learning process here.
       What is AutoML? Read here: https://www.automl.org/automl/

    Attributes
    ----------
    connection_context : hana_ml.dataframe.ConnectionContext
        Connection info to HANA database.
    opt
        Optimizer from pipeline.
    model
        Tuned and fitted HANA PAL model.
    predicted
        Dataframe containing predicted values.
    preprocessor_settings : PreprocessorSettings
        Preprocessor settings.
    """

    def __init__(self, connection_context: hana_ml.dataframe.ConnectionContext):
        self.connection_context = connection_context
        self.opt = None
        self.model = None
        self.predicted = None
        self.preprocessor_settings = None
        self.ensemble = False
        self.columns_to_remove = None
        self.algorithm = None

    def fit(
        self,
        df: pd.DataFrame = None,
        task: str = None,
        steps: int = None,
        target: str = None,
        file_path: str = None,
        table_name: str = None,
        columns_to_remove: list = None,
        categorical_features: list = None,
        id_column=None,
        optimizer: str = "OptunaSearch",
        time_limit=None,
        ensemble=False,
        output_leaderboard=False,
    ):
        """Fits AutoML object

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
            **NOTE**: You must pass whole dataframe, without dividing it in X_train, y_train, etc.
        task: str
            Machine Learning task. 'reg'(regression) and 'cls'(classification) are currently supported.
        steps : int
            Number of iterations.
        target : str
            The column we want to predict. For multiple columns pass a list.
            **Example:** ['feature1', 'feature2'].
        file_path : str
            Path/url to dataframe. Accepts .csv and .xlsx
            **Examples:** 'https://website/dataframe.csv' or 'users/dev/file.csv'
        table_name: str
            Name of table in HANA database
        columns_to_remove: list
            List of columns to delete.
        categorical_features: list
            List of categorical columns. Details here: https://en.wikipedia.org/wiki/Categorical_variable
        id_column: str
            ID column in table. Needed for HANA.
        optimizer: str
            Optimizer to tune hyperparameters.
            Currently supported: "OptunaSearch" (default), "BayesianOptimizer" (unstable)
        time_limit: int
            Amount of time(in seconds) to tune model
        ensemble: str
            Specify if you want to get a blending or stacking ensemble
            Currently supported: "blending", "stacking"
        output_leaderboard : bool
            Print algorithms leaderboard or not
        """
        if time_limit is None and steps is None:
            raise AutoMLError("Specify time limit or number of iterations!")
        if steps is not None:
            if steps < 1:
                raise AutoMLError("The number of steps < 1!")
        if time_limit is not None:
            if time_limit < 1:
                raise AutoMLError("The number of time_limit < 1!")
        inputted = Input(
            connection_context=self.connection_context,
            df=df,
            target=target,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        inputted.load_data()
        if table_name is None:
            table_name = inputted.table_name
        if id_column is None:
            id_column = inputted.id_col
        data = inputted.split_data()
        data.binomial = Preprocessor.check_binomial(
            df=inputted.hana_df, target=data.target
        )
        if columns_to_remove is not None:
            self.columns_to_remove = columns_to_remove
            data.drop(droplist_columns=columns_to_remove)
        pipe = Pipeline(data=data, steps=steps, task=task, time_limit=time_limit)
        self.opt = pipe.train(
            categorical_features=categorical_features, optimizer=optimizer
        )
        if output_leaderboard:
            self.opt.print_leaderboard()
        self.model = self.opt.get_model()
        self.algorithm = self.opt.get_algorithm()
        self.preprocessor_settings = self.opt.get_preprocessor_settings()
        if ensemble and pipe.task == "cls" and not data.binomial:
            raise BlendingError(
                "Sorry, non binomial blending classification is not supported yet!"
            )
        if ensemble:
            if len(self.opt.leaderboard.board) < 3:
                raise BlendingError(
                    "Sorry, not enough fitted models for ensembling! Restart the process"
                )
            print("Starting ensemble accuracy evaluation on the validation data!")
            self.ensemble = ensemble
            if pipe.task == "cls":
                self.model = BlendingCls(
                    categorical_features=categorical_features,
                    id_col=id_column,
                    connection_context=self.connection_context,
                    table_name=table_name,
                    leaderboard=self.opt.leaderboard,
                )
            else:
                self.model = BlendingReg(
                    categorical_features=categorical_features,
                    id_col=id_column,
                    connection_context=self.connection_context,
                    table_name=table_name,
                    leaderboard=self.opt.leaderboard,
                )
            print("\033[33m {}".format("\n"))
            print(
                "Ensemble consists of: "
                + str(self.model.model_list)
                + "\nEnsemble accuracy: "
                + str(self.model.score(data=data))
            )
            print("\033[0m {}".format(""))

    def predict(
        self,
        df: pd.DataFrame = None,
        file_path: str = None,
        table_name: str = None,
        id_column: str = None,
        target_drop: str = None,
    ):
        """Makes predictions using fitted model.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe. **Note**: You must pass whole dataframe, without dividing it in X_train, y_train, etc.
        file_path : str
            Path/url to dataframe. Accepts .csv and .xlsx
        table_name: str
            Name of table in HANA database
        id_column: str
            ID column in table. Needed for HANA.
        target_drop: str
            Target to drop, if it exists in inputted data
        """
        data = Input(
            connection_context=self.connection_context,
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        data.load_data()
        if target_drop is not None:
            data.hana_df = data.hana_df.drop(target_drop)
        if self.columns_to_remove is not None:
            data.hana_df = data.hana_df.drop(self.columns_to_remove)
            print("Columns removed")
        if self.ensemble:
            self.model.id_col = id_column
            self.predicted = self.model.predict(df=data.hana_df, id_colm=id_column)
        else:
            print("Preprocessor settings:", self.preprocessor_settings)
            pr = Preprocessor()
            data.hana_df = pr.clean(
                data=data.hana_df,
                num_strategy=self.preprocessor_settings.tuned_num_strategy,
            )
            self.predicted = self.model.predict(data.hana_df, id_column)
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        print("Prediction results (first 20 rows): \n", res.head(20).collect())
        return res.collect()

    def save_results_as_csv(self, file_path: str):
        """Saves prediciton results to .csv file

        Parameters
        ----------
        file_path : str
            Path to save
        """
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        res.collect().to_csv(file_path)

    def save_stats_as_csv(self, file_path: str):
        """Saves prediciton statistics to .csv file

        Parameters
        ----------
        file_path : str
            Path to save
        """
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[1]
        res.collect().to_csv(file_path)

    def model_to_file(self, file_path: str):
        """Saves model information to JSON file

        Parameters
        ----------
        file_path : str
            Path to save
        """
        with open(file_path, "w+") as file:
            json.dump(self.opt.get_tuned_params(), file)

    def get_algorithm(self):
        """Returns fitted AutoML algorithm"""
        return self.algorithm

    def get_model(self):
        """Returns fitted HANA PAL model"""
        return self.model

    @property
    def optimizer(self):
        """Get optimizer"""
        return self.opt

    @property
    def best_params(self):
        """Get best hyperparameters"""
        return self.opt.get_tuned_params()
