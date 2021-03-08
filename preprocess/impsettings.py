import numpy as np


class ImputerSettings:
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
