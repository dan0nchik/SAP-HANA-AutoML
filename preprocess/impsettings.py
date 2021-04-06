import numpy as np


class ImputerSettings:
    """Settings for preprocessor.

    Attributes
    ----------
    missing_values
        Missing values.
    strategy : str
        Strategy of preprocessing.
    fill_value : str
        Strategy of preprocessing.
    basicvars : str
        Strategy of preprocessing.
    """

    def __init__(
        self, missing_values=np.NaN, strategy="mean", fill_value="0", basicvars="num"
    ):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        if basicvars == "string":
            self.strategy = "most_frequent"
            self.fill_value = "missing_value"
        elif basicvars == "bool":
            self.strategy = "most_frequent"
