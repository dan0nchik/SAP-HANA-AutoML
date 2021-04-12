1. Fitting the model
********************

Ok, you have downloaded the library and ready to do some Data Science! Firstly, let's get used to an example data. It's an imaginary data from bank. We need to predict whether the bank needs to give the loan to client, or not. It's called target feature. In the dataset it is column 'y'.
Let's load the dataframe:

.. code-block:: python

    from automl import dataset
    df = dataset.load_bank()

.. image:: images/bank.png

To connect to HANA database, we need ConnectionContext.
Fill your database credentials there.

.. code-block:: python

    from hana_ml.dataframe import ConnectionContext

    connection_context = ConnectionContext(address='localhost',
                                       user='DEVELOPER',
                                       password='8wGGdQhjwxJtKCYhO5cI3',
                                       port=9999)


Now create the AutoML object. This will be our model.

.. code-block:: python

    model = AutoML(connection_context)
    m.fit(
        table_name="AUTOML505f62ca-1c99-405b-b9d5-8912920038ec",
        df = df,
        target="y",
        id_column='ID',
        categorical_features=["y", 'marital', 'education', 'housing', 'loan'],
        columns_to_remove=['default', 'contact', 'month', 'poutcome'],
        steps=3,
        output_leaderboard=True,
        optimizer="OptunaSearch"
    )

For more information about this, head to :doc:`./automl`.

.. note::
    Pass the **whole** dataframe as *df* parameter. We will automatically divide it in X_train, y_train, etc.


