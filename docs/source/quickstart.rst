Quickstart
**********

Our library in a few lines of code
==================================
.. code-block:: python

      from automl import AutoML
      from hana_ml.dataframe import ConnectionContext

      connection = ConnectionContext('address', port, 'user', 'password')
      model = AutoML(connection)
      model.fit(df=train_df, target='Target', categorical_features=['First', 'Second'])
      model.predict(df=test_df)



