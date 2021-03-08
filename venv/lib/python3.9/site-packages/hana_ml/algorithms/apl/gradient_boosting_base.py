"""
Module defining a common abstract class for SAP HANA APL Gradient Boosting algorithms.
"""
import logging
from collections import OrderedDict
import numpy as np
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.apl.apl_base import APLBase


logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class GradientBoostingBase(APLBase):
    #pylint: disable=too-many-instance-attributes
    """
    Common abstract class for classification and regression.
    """

    def __init__(self,
                 early_stopping_patience,
                 eval_metric,
                 learning_rate,
                 max_depth,
                 max_iterations,
                 conn_context=None,
                 number_of_jobs=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint: disable=too-many-arguments
        super(GradientBoostingBase, self).__init__(
            conn_context,
            variable_storages,
            variable_value_types,
            variable_missing_strings,
            extra_applyout_settings,
            ** other_params)
        self._model_type = None  # to be set by concret subclass
        self._algorithm_name = 'GradientBoosting'
        # --- model params
        self.early_stopping_patience = None
        self.eval_metric = None
        self.learning_rate = None
        self.max_depth = None
        self.max_iterations = None
        self.number_of_jobs = None
        self.label = None
        if early_stopping_patience:
            self.set_params(early_stopping_patience=early_stopping_patience)
        if eval_metric:
            self.set_params(eval_metric=eval_metric)
        if learning_rate:
            self.set_params(learning_rate=learning_rate)
        if max_depth:
            self.set_params(max_depth=max_depth)
        if max_iterations:
            self.set_params(max_iterations=max_iterations)
        if number_of_jobs:
            self.set_params(number_of_jobs=number_of_jobs)

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.

        Parameters
        ----------
        params : dict
            The attribute names and values
        """
        if 'early_stopping_patience' in parameters:
            self.early_stopping_patience = self._arg('early_stopping_patience',
                                                     parameters.pop('early_stopping_patience'),
                                                     int)
        if 'learning_rate' in parameters:
            self.learning_rate = self._arg('learning_rate',
                                           parameters.pop('learning_rate'),
                                           float)
        if 'max_depth' in parameters:
            self.max_depth = self._arg('max_depth',
                                       parameters.pop('max_depth'),
                                       int)
        if 'max_iterations' in parameters:
            self.max_iterations = self._arg('max_iterations',
                                            parameters.pop('max_iterations'),
                                            int)
        if 'number_of_jobs' in parameters:
            self.number_of_jobs = self._arg('number_of_jobs',
                                            parameters.pop('number_of_jobs'),
                                            int)
        if 'label' in parameters:
            self.label = parameters.pop('label')
        return super(GradientBoostingBase, self).set_params(**parameters)

    # pylint: disable=too-many-arguments
    def fit(self, data,
            key=None,
            features=None,
            label=None,
            weight=None):
        """
        Fits the model.

        Parameters
        ----------
        data : DataFrame
            The training dataset
        key : str, optional
            The name of the ID column.
            This column will not be used as feature in the model.
            It will be output as row-id when prediction is made with the model.
            If `key` is not provided, an internal key is created. But this is not the recommended
            usage. See notes below.
        features : list of str, optional
            The names of the features to be used in the model.
            If `features` is not provided, all non-ID and non-label columns will be taken.
        label : str, optional
            The name of the label column. Default is the last column.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        self : object

        Notes
        -----
        It is highly recommended to specify a key column in the training dataset.
        If not, once the model is trained, it won't be possible anymore to have a key defined in
        any input dataset. That is particularly inconvenient to join the predictions output to
        the input dataset.
        """
        if label is None:
            label = data.columns[-1]
        self.label = label
        return self._fit(data=data,
                         key=key,
                         features=features,
                         label=label,
                         weight=weight)

    def get_feature_importances(self): #pylint: disable=unused-argument
        """
        Returns the feature importances.

        Returns
        -------
        feature importances : dict
            { <importance_metric> : OrderedDictionary({ <feature_name> : <value> })
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The INDICATORS table was not found. Please fit the model.")
        df_ind = self.indicators_
        cond = "KEY = 'VariableContribution'"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        df_ind['VALUE'] = df_ind['VALUE'].astype(np.float32)
        df_ind['DETAIL'] = df_ind['DETAIL'].astype(str)  # method name for importances
        # sort by method, value
        df_ind = df_ind.sort_values(['DETAIL', 'VALUE'], ascending=[True, False])
        ret = {}
        for _, row in df_ind.iterrows():
            imp_method = row.loc['DETAIL']
            var_name = row.loc['VARIABLE']
            imp_val = row.loc['VALUE']
            sub_dict = ret.get(imp_method, OrderedDict())
            sub_dict[var_name] = imp_val
            ret[imp_method] = sub_dict
        return ret

    def get_best_iteration(self):
        """
        Returns the iteration that has provided the best performance on the validation dataset
        during the model training.

        Returns
        -------
        The best iteration: int
        """
        if not hasattr(self, 'summary_'):
            raise FitIncompleteError(
                "The SUMMARY table was not found. Please fit the model.")
        cond = "KEY = 'BestIteration'"
        df_s = self.summary_.filter(cond)
        df_s = df_s.collect()  # to pd.DataFrame
        val = df_s['VALUE'].astype(np.int).iloc[0]
        return val

    def get_evalmetrics(self):
        """
        Returns the values of the evaluation metric at each iteration.
        These values are based on the estimation dataset.

        Returns
        -------
        A dictionary:
            {'<MetricName>': <List of values>}
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        cond = "KEY like 'EvalMetric%'"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        df_ind['DETAIL'].fillna(0, inplace=True)
        df_ind['DETAIL'] = df_ind['DETAIL'].astype(np.int)
        df_ind.sort_values(['DETAIL'], inplace=True)
        ret = {}
        vals = []
        metric_name = ''
        for _, row in df_ind.iterrows():
            old_v = row.loc['VALUE']
            detail_v = row.loc['DETAIL']
            if detail_v == 0:
                metric_name = old_v
                continue
            try:
                new_v = float(old_v)
            except ValueError:
                new_v = old_v
            vals.append(new_v)
        ret[metric_name] = vals
        return ret
