import numpy as np


class ImputerSettings:

    def __init__(self, missing_values=np.nan, strategy="mean", fill_value="0", copy=True, basicvars="num"):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        if basicvars == "string":
            self.strategy = "constant"
            self.fill_value = "missing_value"
        elif basicvars == "bool":
            self.strategy = "most_frequent"
