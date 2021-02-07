from sklearn.metrics import accuracy_score, r2_score


class Validate:
    @staticmethod
    def val(algorithm, X_test, y_test, task, metrics=None):
        # TODO get metrics from config (?)
        pred = algorithm.model.predict(X_test)
        # TODO understand model's class to find out right metric
        if task == "cls":
            return accuracy_score(y_test, pred)
        return r2_score(y_test, pred)
