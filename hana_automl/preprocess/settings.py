class PreprocessorSettings:
    """Settings for preprocessor.

    Attributes
    ----------
    num_strategy : str
        Strategy of preprocessing.
    """

    def __init__(self, strategy_by_col: list):
        self.num_strategy: list = ["mean", "median", "delete", "als"]
        self.tuned_num_strategy: str = ""
        self.normalizer_strategy: list = ["min-max", "decimal", "z-score"]
        self.tuned_normalizer_strategy: str = ""
        self.z_score_method: list = ["mean-standard", "mean-mean"]
        self.tuned_z_score_method: str = ""
        self.normalize_int: list = [False, True]
        self.tuned_normalize_int: bool = False
        self.strategy_by_col: list = strategy_by_col
        self.drop_outers: list = [False, True]
        self.tuned_drop_outers: bool = False
        self.categorical_cols: list = None
        self.task: str = None
        self.normalization_exceptions = None
