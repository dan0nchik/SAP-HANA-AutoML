class PreprocessorSettings:
    """Settings for preprocessor.

    Attributes
    ----------
    num_strategy : str
        Strategy of preprocessing.
    """

    def __init__(self, strategy_by_col: list):
        self.num_strategy = ["mean", "median", "delete", "als"]
        self.tuned_num_strategy: str = ""
        self.normalizer_strategy = ["min-max", "decimal"]
        self.tuned_normalizer_strategy: str = ""
        self.z_score_method = ["mean-standard", "mean-mean", "median-median"]
        self.tuned_z_score_method: str = ""
        self.normalize_int = [False]
        self.tuned_normalize_int: bool = False
        self.strategy_by_col = strategy_by_col
