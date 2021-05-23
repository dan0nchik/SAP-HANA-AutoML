class PreprocessorSettings:
    """Settings for preprocessor.

    Attributes
    ----------
    num_strategy : str
        Strategy of preprocessing.
    """

    def __init__(self):
        self.num_strategy = ["mean", "median", "zero"]
        self.tuned_num_strategy: str = ""
        self.normalizer_strategy = ['min-max', 'z-score', 'decimal']
        self.tuned_normalizer_strategy: str = ""
        self.z_score_method = ['mean-standard', 'mean-mean', 'median-median']
        self.tuned_z_score_method: str = ""
        self.normalize_int = [False]
        self.tuned_normalize_int: bool = False

