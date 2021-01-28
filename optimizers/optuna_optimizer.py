from optimizers.base_optimizer import BaseOptimizer
import optuna


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, algo_list, data, problem, iterations, algo_names: list, categorical_features=None,
                 droplist_columns=None):
        super(OptunaOptimizer, self).__init__(algo_list, data, iterations, problem, algo_names=algo_names,
                                              categorical_features=categorical_features,
                                              droplist_columns=droplist_columns)
        opt = optuna.create_study(direction="maximize")
        opt.optimize(self.optobjective, n_trials=iterations)
        self.tuned_params = opt.best_trial
