from cache import SessionState
from hana_automl.automl import AutoML
from hana_automl.utils.connection import connection_context

session_state = SessionState(show_results=False, train=False, automl=AutoML(connection_context))
