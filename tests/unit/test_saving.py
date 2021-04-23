from hana_automl.automl import AutoML
from unittest import mock
import os


@mock.patch("automl.hana_ml.dataframe.DataFrame")
@mock.patch("automl.hana_ml.ConnectionContext")
def test_saving_predicted(conn, hana_df):
    automl = AutoML(conn)
    automl.predicted = hana_df

    automl.save_results_as_csv("path")
    hana_df.collect().to_csv.assert_called_with("path")


@mock.patch("automl.json")
@mock.patch("automl.hana_ml.ConnectionContext")
def test_saving_preprocessor(conn, mock_json):
    automl = AutoML(conn)
    automl.preprocessor_settings = {"imputer": "zero"}
    automl.save_preprocessor("path")
    mock_json.dump.assert_called_once()
    os.remove("path")
