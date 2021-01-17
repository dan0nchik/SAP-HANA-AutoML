class BaseAlgorithm:
    def __init__(self, custom_params: dict = None):
        self.title = ''  # for leaderboard
        self.model = None
        self.params_range = {}
        if custom_params is not None:
            # self.params_range[custom_params.keys()] = custom_params.values()
            pass

    def get_params(self):
        return self.params_range

    def set_params(self, **params):
        self.model.set_params(**params)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, y_test):
        self.model.predict(y_test)
