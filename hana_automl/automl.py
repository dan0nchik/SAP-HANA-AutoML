import json
from typing import Union

import hana_ml
import pandas
import pandas as pd

from hana_automl.algorithms.ensembles.blendcls import BlendingCls
from hana_automl.algorithms.ensembles.blendreg import BlendingReg
from hana_automl.pipeline.data import Data
from hana_automl.pipeline.input import Input
from hana_automl.pipeline.pipeline import Pipeline
from hana_automl.preprocess.preprocessor import Preprocessor
from hana_automl.utils.error import AutoMLError, BlendingError


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

    def __init__(self, connection_context: hana_ml.dataframe.ConnectionContext = None):
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
            df: Union[pandas.DataFrame, hana_ml.dataframe.DataFrame, str] = None,
            file_path: str = None,
            table_name: str = None,
            task: str = None,
            steps: int = None,
            target: str = None,
            columns_to_remove: list = None,
            categorical_features: list = None,
            id_column=None,
            optimizer: str = "OptunaSearch",
            time_limit=None,
            ensemble=False,
            verbosity=2,
            output_leaderboard=False,
            drop_outers=False,
    ):
        """Fits AutoML object

        Parameters
        ----------
        df : pandas.DataFrame or hana_ml.dataframe.DataFrame or str
            **Attention**: You must pass whole dataframe, without dividing it in X_train, y_train, etc.
            See *Notes* for extra information
        file_path : str
            Path/url to dataframe. Accepts .csv and .xlsx. See *Notes* for extra information
            **Examples:** 'https://website/dataframe.csv' or 'users/dev/file.csv'
        table_name: str
            Name of table in HANA database. See *Notes* for extra information
        task: str
            Machine Learning task. 'reg'(regression) and 'cls'(classification) are currently supported.
        steps : int
            Number of iterations.
        target : str
            The column we want to predict. For multiple columns pass a list.
            **Example:** ['feature1', 'feature2'].
        columns_to_remove: list
            List of columns to delete. **Example:** ['column1', 'column2'].
        categorical_features: list
            List of categorical columns. **Example:** ['column1', 'column2'].
            Details here: https://en.wikipedia.org/wiki/Categorical_variable
        id_column: str
            ID column in table. **Example:** 'ID'
            Needed for HANA. If no column passed, it will be created in dataset automatically
        optimizer: str
            Optimizer to tune hyperparameters.
            Currently supported: "OptunaSearch" (default), "BayesianOptimizer" (unstable)
        time_limit: int
            Amount of time(in seconds) to tune the model
        ensemble: bool
            Specify if you want to get a blending or stacking ensemble
            Currently supported: "blending", "stacking"
        verbosity: int
            Level of output. 1 - minimal, 2 - all output.
        output_leaderboard : bool
            Print algorithms leaderboard or not.
        drop_outers: bool
            Try to drop columns outside the base dataset boundaries


        Notes
        -----
        There are multiple options to load data in HANA database. Here are all available parameter combinations: \n
        **1)** df: pandas.DataFrame -> dataframe will be loaded to a new table with random name, like 'AUTOML-9082-842408-12' \n
        **2)** df: pandas.DataFrame + table_name -> dataframe will be loaded to existing table \n
        **3)** df: hana_ml.dataframe.DataFrame -> existing Dataframe will be used \n
        **4)** table_name -> we'll connect to existing table \n
        **5)** file_path -> data from file/url will be loaded to a new table with random name, like 'AUTOML-9082-842408-12' \n
        **6)** file_path + table_name -> data from file/url will be loaded to existing table \n
        **7)** df: str -> we'll connect to existing table \n

        Examples
        --------
        Passing connection info:

        >>> from hana_ml.dataframe import ConnectionContext
        >>> cc = ConnectionContext(address='host',
        ...                        user='user',
        ...                        password='password',
        ...                        port=9999)

        Creating and fitting the model:

        >>> automl = AutoML(cc)
        >>> m.fit(
        ...     df = df,
        ...     target="y",
        ...     id_column='ID',
        ...     categorical_features=["y", 'marital', 'education', 'housing', 'loan'],
        ...     columns_to_remove=['default', 'contact', 'month', 'poutcome'],
        ...     steps=10,
        ... )
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
            verbose=verbosity > 0,
        )
        inputted.load_data()
        if table_name is None:
            table_name = inputted.table_name
        if id_column is None:
            id_column = inputted.id_col
        data = inputted.split_data(categorical_features, drop_outers)
        data.binomial = Preprocessor.check_binomial(
            df=inputted.hana_df, target=data.target
        )
        if columns_to_remove is not None:
            self.columns_to_remove = columns_to_remove
            data.drop(droplist_columns=columns_to_remove)
        pipe = Pipeline(
            data=data,
            steps=steps,
            task=task,
            time_limit=time_limit,
            verbosity=verbosity,
        )
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
            if verbosity > 0:
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
            df: Union[pandas.DataFrame, hana_ml.dataframe.DataFrame, str] = None,
            file_path: str = None,
            table_name: str = None,
            id_column: str = None,
            target_drop: str = None,
            verbosity=1,
    ):
        """Makes predictions using fitted model.

        Parameters
        ----------
        df : pandas.DataFrame or hana_ml.dataframe.DataFrame or str
            **Attention**: You must pass whole dataframe, without dividing it in X_train, y_train, etc.
            See *Notes* for extra information
        file_path : str
            Path/url to dataframe. Accepts .csv and .xlsx. See *Notes* for extra information
            **Examples:** 'https://website/dataframe.csv' or 'users/dev/file.csv'
        table_name: str
            Name of table in HANA database. See *Notes* for extra information
        id_column: str
            ID column in table. Needed for HANA. If None, it will be created in dataset automatically
        target_drop: str
            Target to drop, if it exists in inputted data
        verbosity: int
            Level of output. 1 - minimal, 2 - all output.

        Notes
        -----
        There are multiple options to load data in HANA database. Here are all available parameter combinations: \n
        **1)** df: pandas.DataFrame -> dataframe will be loaded to a new table with random name, like 'AUTOML-9082-842408-12' \n
        **2)** df: pandas.DataFrame + table_name -> dataframe will be loaded to existing table \n
        **3)** df: hana_ml.dataframe.DataFrame -> existing Dataframe will be used \n
        **4)** table_name -> we'll connect to existing table \n
        **5)** file_path -> data from file/url will be loaded to a new table with random name, like 'AUTOML-9082-842408-12' \n
        **6)** file_path + table_name -> data from file/url will be loaded to existing table \n
        **7)** df: str -> we'll connect to existing table \n

        Returns
        -------
        Pandas dataframe with predictions.

        Examples
        --------
        >>> automl.predict(file_path='data/predict.csv',
        ...                table_name='PREDICTION',
        ...                id_column='ID',
        ...                target_drop='target',
        ...                verbosity=1)
        """
        data = Input(
            connection_context=self.connection_context,
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
            verbose=verbosity > 0,
        )
        data.load_data()
        if id_column is None:
            id_column = data.id_col
        if target_drop is not None:
            data.hana_df = data.hana_df.drop(target_drop)
        if self.columns_to_remove is not None:
            data.hana_df = data.hana_df.drop(self.columns_to_remove)
            if verbosity > 0:
                print("Columns removed")
        if self.ensemble:
            self.model.id_col = id_column
            self.predicted = self.model.predict(df=data.hana_df, id_colm=data.id_col)
        else:
            if verbosity > 0:
                print(
                    "Preprocessor settings:",
                    self.preprocessor_settings.tuned_num_strategy,
                )
            pr = Preprocessor()
            data.hana_df = pr.clean(
                data=data.hana_df,
                imputer_num_strategy=self.preprocessor_settings.tuned_num_strategy,
                normalizer_strategy=self.preprocessor_settings.tuned_normalizer_strategy,
                normalizer_z_score_method=self.preprocessor_settings.tuned_z_score_method,
                normalize_int=self.preprocessor_settings.tuned_normalize_int,
            )
            self.predicted = self.model.predict(data.hana_df, data.id_col)
        res = self.predicted
        if type(self.predicted) == tuple:
            res = res[0]
        if verbosity > 0:
            print("Prediction results (first 20 rows): \n", res.head(20).collect())
        return res.collect()

    def score(self, df: Union[pandas.DataFrame, hana_ml.dataframe.DataFrame, str] = None,
              file_path: str = None,
              table_name: str = None,
              target: str = None,
              id_column: str = None):
        inp = Input(
            connection_context=self.connection_context,
            df=df,
            path=file_path,
            table_name=table_name,
            id_col=id_column,
            target=target
        )
        inp.load_data()
        data = Data()
        data.target = inp.target
        data.id_colm = inp.id_col
        if self.ensemble:
            return self.algorithm.score(data)
        else:
            return self.algorithm.score(data, inp.hana_df)

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
