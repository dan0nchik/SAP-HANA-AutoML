3. Fitting the model
********************

Ok, you have downloaded the library and ready to do some Data Science! The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. Column 'y' is the column we will predict.

Let's load the dataframe:

.. code-block:: python

    import pandas as pd #pip install pandas
    df = pd.read_csv('https://raw.githubusercontent.com/dan0nchik/SAP-HANA-AutoML/dev/data/bank_train.csv', index_col='Unnamed: 0')
    df.head()
.. image:: images/bank.png

To connect to HANA database, we need ConnectionContext.
Fill your database credentials there.

.. code-block:: python

    from hana_ml.dataframe import ConnectionContext

    # Replace with your credentials
    connection_context = ConnectionContext(address='database address',
                                           user='your username',
                                           password='your password',
                                           port=39015)

.. tip::
    Store the database credentials securely! For example, put the passwords in a separate config/ini file that is not deployed with the project. 

Now create the AutoML object. This will be our model.

.. code-block:: python
    
    from hana_automl.automl import AutoML
    model = AutoML(connection_context)
    model.fit(
        df = df, # pandas DataFrame, hana_ml dataframe, or "table name".
        target="y", # column to predict
        id_column='ID', # id column
        categorical_features=["y", 'marital', 'education', 'housing', 'loan'],
        columns_to_remove=['default', 'contact', 'month', 'poutcome', 'job'],
        steps=5,
        time_limit=120, # model will train for 2 minutes
        verbose=0
    )

Confused about categorical features? Read about them here. :meth:`hana_automl.automl.AutoML.fit`

.. note::
    Pass the **whole** dataframe as *df* parameter. We will automatically divide it in X_train, y_train, etc.


This is a minimal example. For more advanced usage, head to :meth:`hana_automl.automl.AutoML`
