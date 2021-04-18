from hana_ml.algorithms.pal.neural_network import MLPRegressor

from algorithms.base_algo import BaseAlgorithm


class MLPreg(BaseAlgorithm):
    def __init__(self):
        super(MLPreg, self).__init__()
        self.title = "MLPRegressor"
        self.params_range = {
            "activation": (0, 12),
            "output_activation": (0, 12),
            "hidden_layer_size": (10, 100),
            "normalization": (0, 2),
            "learning_rate": (0.01, 1),
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

    def set_params(self, **params):
        params["hidden_layer_size"] = (int(params["hidden_layer_size"]))
        params["output_activation"] = self.actv[round(params["output_activation"])]
        params["activation"] = self.actv[round(params["activation"])]
        params["normalization"] = ["no", "z-transform", "scalar"][
            round(params["normalization"])
        ]
        params["weight_init"] = [
            "all-zeros",
            "normal",
            "uniform",
            "variance-scale-normal",
            "variance-scale-uniform",
        ][round(params["weight_init"])]
        self.model = MLPRegressor(**params)

    def optunatune(self, trial):
        activation = trial.suggest_categorical("REG_MLP_activation", self.actv)
        output_activation = trial.suggest_categorical(
            "REG_MLP_output_activation", self.actv
        )
        hidden_layer_size = trial.suggest_int(
            "REG_MLP_hidden_layer_size", 10, 100, log=True
        )
        normalization = trial.suggest_categorical(
            "REG_MLP_normalization",
            ["no", "z-transform", "scalar"],
        )
        weight_init = trial.suggest_categorical(
            "REG_MLP_weight_init",
            [
                "all-zeros",
                "normal",
                "uniform",
                "variance-scale-normal",
                "variance-scale-uniform",
            ],
        )
        learning_rate = trial.suggest_float("REG_MLP_learning_rate", 0.01, 1, log=True)
        model = MLPRegressor(
            activation=activation,
            output_activation=output_activation,
            hidden_layer_size=(hidden_layer_size, hidden_layer_size),
            normalization=normalization,
            training_style="batch",
            weight_init=weight_init,
            learning_rate=learning_rate,
        )
        self.model = model
