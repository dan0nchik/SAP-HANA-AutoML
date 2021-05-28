from hana_ml import ConnectionContext
from .credentials import host, user, password, port, dbname

params = {"DATABASENAME": dbname}

connection_context = ConnectionContext(
    address=host, port=port, user=user, password=password, **params
)
