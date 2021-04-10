Quickstart
**********

Our library in 4 lines of code
==============================
.. code-block:: python

      from automl import AutoML
      model = AutoML()
      model.fit(df=df, target='Target', categorical_features=['Name', 'Surname'])
      model.predict(df=test_df)



