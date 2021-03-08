from hana_ml import ConnectionContext
from .credentials import port, host, user, password

connection_context = ConnectionContext(
    address=host, port=port, user=user, password=password
)
