class BaseAlgorithm:
    def __init__(self, custom_params: dict = None):
        self.title = ''
        self.params_range = {}
        if custom_params is not None:
            # self.params_range[custom_params.keys()] = custom_params.values()
            pass
