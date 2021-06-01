4. Predicting
*************
After the model is fitted, we can use it to predict on new data.

.. tip::
    You can pass just a file to the model. We'll extract data from it automatically.

.. code-block:: python
    
    predictions = model.predict(df=pd.read_csv('https://raw.githubusercontent.com/dan0nchik/SAP-HANA-AutoML/main/data/bank_test.csv'),
                                table_name='PREDICTION', # dataset will be loaded in this table in HANA
                                id_column='ID',
                                verbosity=1) # level of output

Again, this is just a code snippet, more information about this function you can access at :meth:`hana_automl.automl.AutoML.predict`

For real-life examples of usage, please visit :doc:`./examples`

