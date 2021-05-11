from unittest import mock

from hana_automl.automl import AutoML


@mock.patch("hana_automl.automl.hana_ml.dataframe.DataFrame")
@mock.patch("hana_automl.automl.hana_ml.ConnectionContext")
def test_saving_predicted(conn, hana_df):
    automl = AutoML(conn)
    automl.predicted = hana_df

    automl.save_results_as_csv("path")
    hana_df.collect().to_csv.assert_called_with("path")
