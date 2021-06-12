5. Saving results
*****************

Prediction results might be saved either in database, or in file.

.. code-block:: python
    
    model.save_results_as_csv('results/predictions.csv')

As predictions are stored in Pandas dataframe, we have to load it to HANA manually.

.. code-block:: python
    
    from hana_ml.dataframe import create_dataframe_from_pandas
    hana_df = create_dataframe_from_pandas(
                connection_context=cc, # connection context from 'Fitting' section
                pandas_df=predictions, # pandas DataFrame from 'Predicting' section
                table_name='SAVED_PREDICTIONS',
            )

Now they are stored in table SAVED_PREDICTIONS, you can work with this table via hana_df variable.

