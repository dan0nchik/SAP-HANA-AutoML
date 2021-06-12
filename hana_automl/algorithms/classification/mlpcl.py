from hana_ml.algorithms.pal.neural_network import MLPClassifier

from hana_automl.algorithms.base_algo import BaseAlgorithm


class MLPcls(BaseAlgorithm):
    def __init__(self):
        super(MLPcls, self).__init__()
        self.title = "MLPClassifier"
        self.params_range = {
            "activation": (0, 12),
            "output_activation": (0, 12),
            "hidden_layer_size": (1, 3),
            "normalization": (0, 2),
            "learning_rate": (1e-4, 0.5),
            "weight_init": (0, 4),
        }
        self.actv = [
            "tanh",
            "linear",
            "sigmoid_asymmetric",
            "sigmoid_symmetric",
            "gaussian_asymmetric",
            "gaussian_symmetric",
            "elliot_asymmetric",
            "elliot_symmetric",
            "sin_asymmetric",
            "sin_symmetric",
            "cos_asymmetric",
            "cos_symmetric",
            "relu",
        ]
        self.weight_init = [
            "all-zeros",
            "normal",
            "uniform",
            "variance-scale-normal",
            "variance-scale-uniform",
        ]

    def set_params(self, **params):
        params["output_activation"] = self.actv[round(params["output_activation"])]
        params["activation"] = self.actv[round(params["activation"])]
        # TypeError: Parameter 'hidden_layer_size' must be type of tuple/list of int.
        params["hidden_layer_size"] = int(params["hidden_layer_size"])
        params["normalization"] = ["no", "z-transform", "scalar"][
            round(params["normalization"])
        ]
        params["training_style"] = "batch"
        params["weight_init"] = self.weight_init[round(params["weight_init"])]
        # self.model = UnifiedClassification(func='MLP', **params)
        self.tuned_params = params
        self.model = MLPClassifier(**params)

    def optunatune(self, trial):
        activation = trial.suggest_categorical("activation", self.actv)
        output_activation = trial.suggest_categorical("output_activation", self.actv)
        hidden_layer_size = trial.suggest_int("hidden_Layer_Size", 1, 3, log=True)
        normalization = trial.suggest_categorical(
            "normalization",
            ["no", "z-transform", "scalar"],
        )
        weight_init = trial.suggest_categorical(
            "weight_init",
            self.weight_init,
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.5, log=True)
        """
        model = UnifiedClassification(
            func='MLP',
            activation=activation,
            output_activation=output_activation,
            hidden_layer_size=(hidden_layer_size, hidden_layer_size),
            normalization=normalization,
            training_style="batch",
            weight_init=weight_init,
            learning_rate=learning_rate,
        )"""
        model = MLPClassifier(
            activation=activation,
            output_activation=output_activation,
            hidden_layer_size=(hidden_layer_size, hidden_layer_size),
            normalization=normalization,
            training_style="batch",
            weight_init=weight_init,
            learning_rate=learning_rate,
        )
        self.model = model
