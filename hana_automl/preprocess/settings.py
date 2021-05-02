class PreprocessorSettings:
    """Settings for preprocessor.

    Attributes
    ----------
    num_strategy : str
        Strategy of preprocessing.
    """

    def __init__(self):
        self.num_strategy = ["mean", "median", "zero"]
        self.tuned_num_strategy: str = ''
