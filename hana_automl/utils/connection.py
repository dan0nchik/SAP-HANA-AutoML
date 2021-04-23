from hana_ml import ConnectionContext
from hana_automl.utils.credentials import port, host, user, password, dbname

params = {"DATABASENAME": dbname}

connection_context = ConnectionContext(
    address=host, port=port, user=user, password=password, **params
)
