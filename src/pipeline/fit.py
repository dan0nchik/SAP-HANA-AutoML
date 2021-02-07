class Fit:
    @staticmethod
    def fit(model, X_train, y_train):
        model.fit(X_train, y_train.values.ravel())
