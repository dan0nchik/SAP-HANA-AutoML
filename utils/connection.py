from hana_ml import ConnectionContext
from utils.credentials import port, host, user, password

params = {"DATABASENAME": "HXE"}

connection_context = ConnectionContext(
    address=host, port=port, user=user, password=password, **params
)
