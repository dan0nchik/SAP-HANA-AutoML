from hana_ml import ConnectionContext
from .credentials import host, user, password, port, schema

schema = schema
connection_context = ConnectionContext(
    address=host, port=port, user=user, password=password
)
