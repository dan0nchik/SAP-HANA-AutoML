3. Predicting
*************
After the model is fitted, we can use it to predict on new data.

.. tip::
    You can pass just a file to the model. We'll extract data from it automatically.

.. code-block:: python
    
    predictions = model.predict(file_path='data/predict.csv',
                                table_name='PREDICTION',
                                target_drop='y',
                                verbosity=1)

Again, this is just a code snippet, more information about this function you can access :doc:`./automl`

For real-life examples of usage, please visit :doc:`./examples`

