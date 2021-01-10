from preprocessor import Preprocessor


class Pipeline:
    # TODO add x_test and y_test if needed
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        pass

    def start(self):
        # TODO: write & start optimizer here
        pr = Preprocessor(['basic params from optimizer'])

        pass
