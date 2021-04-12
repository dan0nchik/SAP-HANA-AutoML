import pandas as pd
import json
from pipeline.input import Input
from pipeline.pipeline import Pipeline
from preprocess.preprocessor import Preprocessor
from utils.error import AutoMLError
from hana_ml.model_storage import ModelStorage
import pickle
import hana_ml


class AutoML:
    """Main class. Control the whole Automated Machine Learning process here.
       What is AutoML? Read here: https://www.automl.org/automl/

    Attributes
    ----------
    connection_context : hana_ml.ConnectionContext
        Connection info to HANA database.
    opt
        Optimizer from pipeline.
    model
        Tuned and fitted HANA PAL model.
    predicted
        Dataframe containig predicted values.
    preprocessor_settings : dict
        Preprocessor settings.
    """

    def __init__(self, connection_context: hana_ml.ConnectionContext):
        self.connection_context = connection_context
        self.opt = None
        self.model = None
        self.predicted = None
        self.preprocessor_settings = None

    def fit(
        self,
        df: pd.DataFrame = None,
        steps: int = 10,
        target: str = None,
        file_path: str = None,
        table_name: str = None,
        columns_to_remove: list = None,
        categorical_features: list = None,
        id_column=None,
        optimizer: str = "OptunaSearch",
        config=None,
        output_leaderboard=False,
    ):
        """Fits AutoML object

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
            **NOTE**: You must pass whole dataframe, without dividing it in X_train, y_train, etc.
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
        config : dict
            Configuration file (not implemented yet)
        output_leaderboard : bool
            Print algorithms leaderboard or not
        """
        if steps < 1:
            raise AutoMLError("The number of steps < 1!")
        inputted = Input(
            connection_context=self.connection_context,
            df=df,
            target=target,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        inputted.load_data()
        data = inputted.split_data()
        if columns_to_remove is not None:
            data.drop(droplist_columns=columns_to_remove)
        pipe = Pipeline(data, steps)
        self.opt = pipe.train(
            categorical_features=categorical_features, optimizer=optimizer
        )
        if output_leaderboard:
            self.opt.print_leaderboard()
        self.model = self.opt.get_model()
        self.preprocessor_settings = self.opt.get_preprocessor_settings()

    def predict(
        self,
        df: pd.DataFrame = None,
        file_path: str = None,
        table_name: str = None,
        id_column: str = None,
        preprocessor_file: str = None,
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
        preprocessor_file : str
            Path to JSON file containing preprocessor settings.
        """
        data = Input(
            connection_context=self.connection_context,
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
        )
        data.load_data()

        if preprocessor_file is not None:
            with open(preprocessor_file) as json_file:
                json_data = json.load(json_file)
                self.preprocessor_settings = json_data
        print("Preprocessor settings:", self.preprocessor_settings)
        pr = Preprocessor()
        data.hana_df = pr.clean(
            data=data.hana_df, num_strategy=self.preprocessor_settings["imputer"]
        )
        self.predicted = self.model.predict(data.hana_df, id_column)
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        print("Prediction results (first 20 rows): \n", res.head(20).collect())

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

    def save_preprocessor(self, file_path: str):
        """Saves preprocessor settings to JSON file to use it in future predictions.

        Parameters
        ----------
        file_path : str
            Path to save
        """
        with open(file_path, "w+") as file:
            json.dump(self.preprocessor_settings, file)

    def model_to_file(self, file_path: str):
        """Saves model information to JSON file

        Parameters
        ----------
        file_path : str
            Path to save
        """
        with open(file_path, "w+") as file:
            json.dump(self.opt.get_tuned_params(), file)

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


class Storage(ModelStorage):
    """Save your HANA PAL model easily.
    Details here:
    https://help.sap.com/doc/1d0ebfe5e8dd44d09606814d83308d4b/2.0.04/en-US/hana_ml.model_storage.html"""

    def __init__(self, connection_context):
        super().__init__(connection_context)
