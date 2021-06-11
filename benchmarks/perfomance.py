import hana_ml
import time
import matplotlib.pyplot as plt
import pandas as pd
from hana_ml.algorithms.apl.classification import AutoClassifier
from hana_ml.algorithms.apl.gradient_boosting_classification import (
    GradientBoostingBinaryClassifier,
)
from hana_ml.algorithms.apl.gradient_boosting_classification import (
    GradientBoostingClassifier,
)
from hana_ml.algorithms.apl.gradient_boosting_regression import (
    GradientBoostingRegressor,
)
from hana_ml.algorithms.apl.regression import AutoRegressor
from hana_ml.dataframe import create_dataframe_from_pandas
from sklearn.model_selection import train_test_split
from benchmarks.cleanup import clean
from hana_automl.automl import AutoML


class Benchmark:
    def __init__(
        self, connection_context: hana_ml.dataframe.ConnectionContext, schema: str
    ):
        self.connection_context = connection_context
        self.apl_model = None
        self.automl_model = None
        self.apl_accuracy = 0
        self.automl_accuracy = 0
        self.schema = schema

    def run(
        self,
        dataset: str,
        task: str,
        grad_boost: bool,
        label: str,
        id_column: str = None,
        categorical: list = None,
    ):
        df = pd.read_csv(dataset)
        ensemble: bool = False
        metric = None

        if id_column is None:
            df["ID"] = range(0, len(df))
            id_column = "ID"

        train, test = train_test_split(df, test_size=0.3)

        train_df = create_dataframe_from_pandas(
            self.connection_context,
            table_name="BENCHMARK_TRAIN",
            pandas_df=train,
            force=True,
            drop_exist_tab=True,
        )
        train_df.declare_lttab_usage(True)
        test_df = create_dataframe_from_pandas(
            self.connection_context,
            table_name="BENCHMARK_TEST",
            pandas_df=test,
            force=True,
            drop_exist_tab=True,
        )
        test_df.declare_lttab_usage(True)

        if task == "cls":
            if grad_boost:
                self.apl_model = GradientBoostingClassifier(self.connection_context)
            else:
                self.apl_model = AutoClassifier(self.connection_context)

        if task == "binary_cls":
            if grad_boost:
                self.apl_model = GradientBoostingBinaryClassifier(
                    self.connection_context
                )
            else:
                self.apl_model = AutoClassifier(self.connection_context)
        if task == "reg":
            ensemble = True
            metric = "mae"
            if grad_boost:
                self.apl_model = GradientBoostingRegressor(self.connection_context)
            else:
                self.apl_model = AutoRegressor(self.connection_context)

        self.automl_model = AutoML(self.connection_context)
        start_time = time.time()
        self.apl_model.fit(train_df, key=id_column, label=label)
        print(f"Finished in {round(time.time() - start_time)} seconds")
        self.apl_accuracy = self.apl_model.score(test_df)
        print("APL accuracy: ", self.apl_accuracy)
        clean(self.connection_context, self.schema)
        start_time = time.time()

        self.automl_model.fit(
            df,
            table_name="BENCHMARK_AUTOML_TABLE",
            steps=10,
            target=label,
            categorical_features=categorical,
            id_column=id_column,
            task=task,
            verbose=0,
            ensemble=ensemble,
            tuning_metric=metric,
        )
        print(f"Finished in {round(time.time() - start_time)} seconds")
        self.automl_accuracy = self.automl_model.accuracy
        print("hana_automl accuracy:", self.automl_accuracy)

    def plot_results(self):
        plt.barh(["hana_automl", "APL"], [self.automl_accuracy, self.apl_accuracy])
        plt.xlabel("Accuracy")
        plt.ylabel("Models")
        plt.show()
