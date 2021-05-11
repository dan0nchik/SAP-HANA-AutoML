from .cache import SessionState
from hana_automl.automl import AutoML

session_state = SessionState(show_results=False, train=False, automl=AutoML(None))
