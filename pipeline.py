from preprocessor import Preprocessor


class Pipeline:
    # TODO add x_test and y_test if needed
    def __init__(self, X_train, y_train, X_test=None, y_test=None, iterations=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.iter = iterations
        pass

    def train(self):
        # TODO: write & start optimizer here
        pr = Preprocessor()
        # rewrite bad code
        self.X_train = pr.clean(self.X_train)
        self.y_train = pr.clean(self.y_train)
        self.X_test = pr.clean(self.X_test)
        self.y_test = pr.clean(self.y_test)

        # for i in range(0, self.iter):
        # TODO fit, validate, optimize
        # output

