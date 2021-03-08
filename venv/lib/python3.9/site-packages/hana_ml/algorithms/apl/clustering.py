#pylint: disable=too-many-lines
"""
This module provides the SAP HANA APL clustering algorithms.

The following classes are available:

    * :class:`AutoUnsupervisedClustering`
    * :class:`AutoSupervisedClustering`
"""
from collections import OrderedDict
import re
import logging
import numpy as np
import pandas as pd
from hana_ml.dataframe import (
    DataFrame,
    quotename)
from hana_ml.ml_exceptions import FitIncompleteError, Error
from hana_ml.algorithms.apl.apl_base import (
    APLBase,
    APLArtifactTable)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name


class _AutoClusteringBase(APLBase):
    """
    Abstract class for the SAP HANA APL Clustering algorithm.
    """
    APL_ALIAS_KEYS = {
        'model_type': 'APL/ModelType',
        'calculate_cross_statistics': 'APL/CalculateCrossStatistics',
        'calculate_sql_expressions': 'APL/CalculateSQLExpressions',
        'cutting_strategy': 'APL/CuttingStrategy',
        'encoding_strategy': 'APL/EncodingStrategy',
    }

    def __init__(self,
                 conn_context=None,
                 nb_clusters=None,
                 nb_clusters_min=None,
                 nb_clusters_max=None,
                 distance=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint: disable=too-many-arguments
        super(_AutoClusteringBase, self).__init__(
            conn_context=conn_context,
            variable_storages=variable_storages,
            variable_value_types=variable_value_types,
            variable_missing_strings=variable_missing_strings,
            extra_applyout_settings=extra_applyout_settings,
            ** other_params)
        self._model_type = 'clustering'
        # --- model params
        self.nb_clusters = self._arg('nb_clusters', nb_clusters, int)
        self.nb_clusters_min = self._arg('nb_clusters_min', nb_clusters_min, int)
        self.nb_clusters_max = self._arg('nb_clusters_max', nb_clusters_max, int)
        if distance:
            self.set_params(distance=distance)
        else:
            self.distance = None

    def _create_train_config_table(self):
        """
        Creates a new APLArtifactTable object for the "TRAIN_OPERATION_LOG" table.

        Returns
        -------
        An APLArtifactTable object with data
        """
        if self._model_type is None:
            raise FitIncompleteError("Model type undefined.")
        train_config_ar = [(self.APL_ALIAS_KEYS['model_type'],
                            self._model_type, None)]
        # add params to OPERATION_CONFIG table
        if self.nb_clusters:
            train_config_ar.append(('APL/NbClusters', str(self.nb_clusters), None))
        if self.nb_clusters_min:
            train_config_ar.append(('APL/NbClustersMin', str(self.nb_clusters_min), None))
        if self.nb_clusters_max:
            train_config_ar.append(('APL/NbClustersMax', str(self.nb_clusters_max), None))
        if self.distance:
            train_config_ar.append(('APL/Distance', str(self.distance), None))
        # add other params to OPERATION_CONFIG table
        train_config_ar = train_config_ar + self._get_train_config_data()
        train_config_df = pd.DataFrame(train_config_ar)
        train_config_table = self._create_aplartifact_table_with_data_frame(
            name='#CREATE_AND_TRAIN_CONFIG_{}'.format(self.id),
            type_name=APLArtifactTable.OPERATION_CONFIG_EXTENDED,
            data_df=train_config_df)
        return train_config_table

    # pylint: disable=too-many-arguments
    def _fit(self, data, key=None, features=None, label=None, weight=None):
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
            If `key` is not provided, an internal key is created. But this is not recommended
            usage. See notes below.
        features : list of str, optional
            The names of the features to be used in the model.
            If `features` is not provided, all non-ID and non-label columns will be taken.
        label : str, optional
            label (target variable).
            It must be provided only for supervised clustering.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        self : object

        Notes
        -----
        It is highly recommended to use a dataset with key in the fit() method.
        If not, once the model is trained, it will not be possible anymore to
        use the predict() method with a dataset with key, because the model will not expect it.
        """
        if not features:
            features = [col for col in data.columns if col not in [key, label, weight]]
        return super(_AutoClusteringBase, self)._fit(
            data=data,
            key=key,
            features=features,
            label=label,
            weight=weight,
        )

    def predict(self, data):
        """
        Predicts which cluster each specified row belongs to.

        Parameters
        ----------
        data : hana_ml DataFrame
            The set of rows for which to generate cluster predictions.
            This dataset must have the same structure as the one used in the fit() method.

        Returns
        -------
        hana_ml DataFrame
            By default, the ID of the closest cluster and the distance to its center are provided.
            Users can request different outputs by setting the **extra_applyout_settings** parameter
            in the model.
            The **extra_applyout_settings** parameter is a dictionary with **'mode'** and
            **'nb_distances'** as keys.
            If **mode** is set to **'closest_distances'**, cluster IDs and distances to centroids
            will be provided from the closest to the furthest cluster.
            The output columns will be:
                - <The key column name>,
                - CLOSEST_CLUSTER_1,
                - DISTANCE_TO_CLOSEST_CENTROID_1,
                - CLOSEST_CLUSTER_2,
                - DISTANCE_TO_CLOSEST_CENTROID_2,
                ...
            If **mode** is set to **'all_distances'**, the distances to the centroids of all
            clusters will be provided in cluster ID order. The output columns will be:
                - ID,
                - DISTANCE_TO_CENTROID_1,
                - DISTANCE_TO_CENTROID_2,
                ...
            **nb_distances** limits the output to the closest clusters. It is only valid when
            **mode** is **'closest_distances'** (it will be ignored if **mode** = 'all distances').
            It can be set to **'all'** or a positive integer.

        Examples
        --------

        Retrieves the IDs of the 3 closest clusters and the distances to their centroids:

        >>> extra_applyout_settings = {'mode': 'closest_distances', 'nb_distances': '3'}
        >>> model.set_params(extra_applyout_settings=extra_applyout_settings)
        >>> out = model.predict(hana_df)
        >>> out.head(3).collect()
                    id  CLOSEST_CLUSTER_1  ...  CLOSEST_CLUSTER_3  DISTANCE_TO_CLOSEST_CENTROID_3
        0   30                  3  ...                  4                        0.730330
        1   63                  4  ...                  1                        0.851054
        2   66                  3  ...                  4                        0.730330

        Retrieves the distances to all clusters:

        >>> model.set_params(extra_applyout_settings={'mode': 'all_distances'})
        >>> out = model.predict(hana_df)
        >>> out.head(3).collect()
           id  DISTANCE_TO_CENTROID_1  DISTANCE_TO_CENTROID_2  ... DISTANCE_TO_CENTROID_5
        0  30                0.994595                0.877414  ...              0.782949
        1  63                0.994595                0.985202  ...              0.782949
        2  66                0.994595                0.877414  ...              0.782949
        """
        if self.extra_applyout_settings is None:
            mode = 'closest_distances'
        else:
            mode = self.extra_applyout_settings.get('mode', 'closest_distances')
        # Determines the number of distances to be generated in output
        if self.extra_applyout_settings is None:
            nb_distances = '1'  # distance from the closest cluster
        else:
            if mode == 'closest_distances':
                nb_distances = self.extra_applyout_settings.get('nb_distances', '1')
            else:
                nb_distances = 'all'
        return self._predict_clusters(data, mode=mode, nb_distances=nb_distances)

    def _predict_clusters(self, data, mode, nb_distances):
        """
        Abstract method, it is not exposed to the end user.
        It will be overridden by the subclasses.
        """
        raise NotImplementedError()

    def _finalize_applyout_df(self, key_col, max_cluster_no, map_new_cols):
        """
        Creates a hana_ml dataframe according to extra_output_settings.
        It is used by the predict() method to rewrite the output in a proper form.

        Returns
        ------
        key_col: str
            The key column name
        max_cluster_no : int
            The max cluster id actually found by the algorithm
        map_new_cols : dict
            {NEW_NAME : OLD_NAME}, the mapping between the old column names and the new ones.

        """
        sql = 'SELECT '
        if key_col:
            # always put key as first column
            sql = sql + quotename(key_col)
        for i in range(1, max_cluster_no + 1):
            # order columns by cluster
            # all columns ending with '_<i>'
            s_end = '_' + str(i)
            cols = [col for col in map_new_cols.keys() if col.endswith(s_end)]
            cols = sorted(cols)
            for col in cols:
                sql = sql + ', '
                sql = (sql + '{old_col} {new_col}'.format(
                    old_col=quotename(map_new_cols[col]),
                    new_col=quotename(col)))

        sql = sql + ' FROM ' + self.applyout_table_.name
        applyout_df_new = DataFrame(connection_context=self.conn_context,
                                    select_statement=sql)
        logger.info('DataFrame for predict ouput: %s', sql)
        return applyout_df_new

    def _get_metrics(self, cond):
        """
        Returns a dictionary containing the metrics on the clustering quality.

        Parameters
        ----------
        cond : str
            The condition clause to be put into the SELECT statement on the "INDICATORS" table.
            It varies depending on whether the clustering is supervised or unsupervised.

        Returns
        -------
        A dictionary with metric name as key and metric value as value.
        For example:
        {'SimplifiedSilhouette': 0.14668968897882997,
         'RSS': 24462.640041325714,
         'IntraInertia': 3.2233573348587714}
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        # Creates the OrderedDict to be returned
        # Converts df_ind.VALUE str to float if possible
        ret = {}
        for key, old_v in zip(df_ind.KEY, df_ind.VALUE):
            try:
                new_v = float(old_v)
            except ValueError:
                new_v = old_v
            ret[key] = new_v
        return ret

    def _get_metrics_by_cluster(self, cond, label=None):
        """
        Returns a dictionary containing various metrics for each cluster.

        Parameters
        ----------
        cond : str
            The condition to be put into the SELECT statement on the "INDICATORS" table
        label: str
            The label column name

        Returns
        -------
        A nested dictionary: {'metric_name' : {'cluster_id': value}}

        Example
        -------
        The model has 3 clusters:

        >>> model.get_metrics_by_cluster()
        {'Frequency': {1: 0.27619047619047621,
          2: 0.48571428571428571,
          3: 0.23809523809523808},
         'IntraInertia': {1: 0.117988390733077,
          2: 0.24933321375486531,
          3: 0.20872319376319992},
         'RSS': {1: 3.4216633312592331,
          2: 12.71599390149813,
          3: 5.2180798440799983},
         'SimplifiedSilhouette': {1: 0.66753523720280605,
          2: 0.51720585645064321,
          3: 0.4381581807252532}}
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        df_ind.rename({'DETAIL': 'CLUSTER'}, axis=1, inplace=True)  # rename column
        df_ind['CLUSTER'] = df_ind['CLUSTER'].astype('int32')  # change str to int
        # pivot table
        df_metrics = df_ind.pivot(index='CLUSTER', columns='KEY', values='VALUE')
        # remove 'Cluster' from column names
        df_metrics.columns = [col[7:] for col in df_metrics.columns]
        # Drop column label (for supervised clustering)
        if label:
            if label in df_metrics.columns:
                df_metrics.drop(label, inplace=True)
        df_metrics = df_metrics.astype(float)  # converts to float
        # transform dataframe to nested dictionaries for the final output
        dict_metrics = {}
        for metric_name in df_metrics.columns:
            dict_metrics[metric_name] = dict(df_metrics[metric_name])
        # add KL metrics
        dict_metrics['KL'] = self._get_cluster_kl()
        return dict_metrics

    def _get_cluster_kl(self):
        """
        Returns a dictionary containing the Kullback-Leibler (KL) divergence between the cluster
        population and the overall population.
        The higher the KL of a cluster, the more informative it will be.

        Returns
        -------
        A nested dictionary: {'cluster_id' : OrderedDictionary}
        where OrderedDictionary is {feature_name : KL_value}
        The feature names are ordered in descending order of KL_value.

        Example
        -------
        >>> model._get_cluster_kl()
        {1: OrderedDict([('relationship', 0.684001270611), ('education', 0.675109873839),...
         2: OrderedDict([('relationship', 0.6088910048596), ('marital-status', 0.5453827),...
        ...
        """
        if not hasattr(self, 'indicators_'):
            raise FitIncompleteError(
                "The indicators table was not found. Please fit the model.")
        df_ind = self.indicators_
        cond = "KEY='ClusterKL'"
        df_ind = df_ind.filter(cond)   # hana DataFrame
        df_ind = df_ind.collect()  # to pd.DataFrame
        df_ind = df_ind[['DETAIL', 'VARIABLE', 'VALUE']]

        df_ind.DETAIL = df_ind.DETAIL.astype(int)  # cluster_id
        df_ind.VALUE = df_ind.VALUE.astype(float)
        d_ret = {}
        for cluster_id in np.sort(df_ind.DETAIL.unique()):
            sub_set = df_ind[df_ind.DETAIL == cluster_id].sort_values('VALUE', ascending=False)
            feat_kl = []
            for _, row in sub_set.iterrows():
                feat = row['VARIABLE']
                kl_val = row['VALUE']
                feat_kl.append((feat, kl_val))
            d_ret[cluster_id] = OrderedDict(feat_kl)
        return d_ret

    def set_params(self, **parameters):
        """
        Sets attributes of the current model.

        Parameters
        ----------
        params : dictionary
            The set of parameters with their new values
        """
        # Checks the validity of the parameter extra_applyout_settings
        if 'extra_applyout_settings' in parameters:
            # It must be a dictionary with key = 'mode' or 'nb_distances'
            param = parameters['extra_applyout_settings']
            if param:
                if not isinstance(param, dict):
                    raise TypeError("'extra_applyout_settings' must be a dictionary")
                if not set(param.keys()).issubset(['mode', 'nb_distances']):
                    msg = ("The keys of extra_applyout_settings must be either "
                           + "'mode' or 'nb_distances'")
                    raise KeyError(msg)
                # Checks the mode is either 'closest_distances' or 'nb_distances'
                # Checks 'nb_distances' is either 'all' or an integer
                mode = param.get('mode', 'closest_distances')
                if mode not in ['closest_distances', 'all_distances']:
                    msg = ("The 'mode' parameter must be either "
                           + "'closest_distances' or 'all_distances'")
                    raise ValueError(msg)
                nb_dist = str(param.get('nb_distances', 1))
                if (nb_dist != 'all') and (not nb_dist.isdigit()):
                    msg = ("The 'nb_distances' parameter must be either "
                           + "'all' or an integer")
                    raise ValueError(msg)
        if 'distance' in parameters:
            param = self._arg('distance',
                              parameters.pop('distance'),
                              str)
            if param not in ['L1', 'L2', 'LInf', 'SystemDetermined']:
                raise ValueError("The 'distance' parameter must be in "
                                 "['L1', 'L2', 'LInf', 'SystemDetermined']")
            self.distance = param
        return super(_AutoClusteringBase, self).set_params(**parameters)


class AutoUnsupervisedClustering(_AutoClusteringBase):
    """
    SAP HANA APL unsupervised clustering algorithm.

    Parameters
    ----------
    nb_clusters : int, optional, default = 10
        The number of clusters to create
    nb_clusters_min: int, optional
        The minimum number of clusters to create.
        If the nb_clusters parameter is set, it will be ignored.
    nb_clusters_max: int, optional
        The maximum number of clusters to create.
        If the nb_clusters parameter is set, it will be ignored.
    distance: str, optional, default = 'SystemDetermined'
        The metric used to measure the distance between data points.
        The possible values are: 'L1', 'L2', 'LInf', 'SystemDetermined'.
    variable_storages: dict, optional
        Specifies the variable data types (string, integer, number).
        For example, {'VAR1': 'string', 'VAR2': 'number'}.
        See notes below for more details.
    variable_value_types: dict, optional
        Specifies the variable value types (continuous, nominal, ordinal).
        For example, {'VAR1': 'continuous', 'VAR2': 'nominal'}.
        See notes below for more details.
    variable_missing_strings: dict, optional
        Specifies the variable values that will be taken as missing.
        For example, {'VAR1': '???'} means anytime the variable value equals '???',
        it will be taken as missing.
    extra_applyout_settings: dict optional
        Defines the output to generate when applying the model.
        See documentation on predict() method for more information.
    other_params: dict optional
        Corresponds to the advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
            - calculate_cross_statistics
            - calculate_sql_expressions
            - cutting_strategy
            - encoding_strategy
        See *Common APL Aliases for Model Training* in the `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_train_apl_aliases: dict, optional
        Users can provide APL aliases as advanced settings to the model.
        Unlike 'other_params' described above, users are free to input any possible value in
        'other_train_apl_aliases'. There is no control in python.

    Attributes
    ----------
    model_ : hana_ml DataFrame
        The trained model content
    summary_ : APLArtifactTable
        The reference to the "SUMMARY" table generated by the model training.
        This table contains the summary about the model training.
    indicators_ : APLArtifactTable
        The reference to the "INDICATORS" table generated by the model training.
        This table contains the various metrics related to the model and its variables.
    fit_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table generated by the model training
    var_desc_ : APLArtifactTable
        The reference to the "VARIABLE_DESCRIPTION" table that was built during the model training
    applyout_ : hana_ml DataFrame
        The predictions generated the last time the model was applied
    predict_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table when a prediction was made

    Notes
    -----
        - The algorithm may detect less clusters than requested.
        This happens when a cluster detected on the estimation dataset was not found on
        the validation dataset. In that case, this cluster will be considered unstable and will
        then be removed from the model.
        Users can get the number of clusters actually found in the "INDICATORS" table. For example,
        ::
            # The actual number of clusters found
            d = model_u.get_indicators().collect()
            d[d.KEY=='FinalClusterCount'][['KEY','VALUE']]

        - It is highly recommended to use a dataset with a key provided in the fit() method.
        If not, once the model is trained, it will not be possible anymore to
        use the predict() method with a key, because the model will not expect it.

        - By default, when it is not given, SAP HANA APL guesses the variable description by reading
        the first 100 rows. But, sometimes, it does not provide the correct result.
        By specifically providing values for these parameters, the user can overwrite the default
        guess. For example:
        ::
            model.set_params(
                    variable_storages = {
                        'ID': 'integer',
                        'sepal length (cm)': 'number'
                        })
            model.set_params(
                    variable_value_types = {
                        'sepal length (cm)': 'continuous'
                        })
            model.set_params(
                    variable_missing_strings = {
                        'sepal length (cm)': '-1'
                        })

    Examples
    --------
    >>> from hana_ml.algorithms.apl.clustering import AutoUnsupervisedClustering
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA

    >>> CONN = ConnectionContext('HDB_HOST', HDB_PORT, 'HDB_USER', 'HDB_PASS')
    >>> # -- Creates Hana DataFrame
    >>> hana_df = DataFrame(CONN, 'select * from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = AutoUnsupervisedClustering(CONN, nb_clusters=5)
    >>> model.fit(data=hana_df, key='id')

    Debriefing

    >>> model.get_metrics()
    OrderedDict([('SimplifiedSilhouette', 0.3448029020802121), ('RSS', 4675.706587754118),...

    >>> model.get_metrics_by_cluster()
    {'Frequency': {1: 0.23053242076908276,
          2: 0.27434649954646656,
          3: 0.09628652318517908,
          4: 0.29919463456199663,
          5: 0.09963992193727494},
         'IntraInertia': {1: 0.6734978174937322,
          2: 0.7202839995396123,
          3: 0.5516800856975772,
          4: 0.6969632183111357,
          5: 0.5809322138167139},
         'RSS': {1: 5648.626195319932,
          2: 7189.15459940487,
          3: 1932.5353401986129,
          4: 7586.444631316713,
          5: 2105.879275085588},
         'SimplifiedSilhouette': {1: 0.1383827622819234,
          2: 0.14716862328457128,
          3: 0.18753797605134545,
          4: 0.13679980173383793,
          5: 0.15481377834381388},
         'KL': {1: OrderedDict([('relationship', 0.4951910610641741),
                       ('marital-status', 0.2776259711735807),
                       ('hours-per-week', 0.20990189265572687),
                       ('education-num', 0.1996353893520096),
                       ('education', 0.19963538935200956),
                       ...

    Predicting which cluster a data point belongs to

    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  CLOSEST_CLUSTER_1  DISTANCE_TO_CLOSEST_CENTROID_1
    0   30                  3                        0.640378
    1   63                  4                        0.611050
    2   66                  3                        0.640378
    3  110                  4                        0.611050
    4  335                  1                        0.851054

    Determining the 2 closest clusters

    >>> model.set_params(extra_applyout_settings={'mode':'closest_distances', 'nb_distances': 2})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  CLOSEST_CLUSTER_1  ...  CLOSEST_CLUSTER_2  DISTANCE_TO_CLOSEST_CENTROID_2
    0   30                  3  ...                  4                        0.730330
    1   63                  4  ...                  1                        0.851054
    2   66                  3  ...                  4                        0.730330
    3  110                  4  ...                  1                        0.851054
    4  335                  1  ...                  4                        0.906003

    Retrieving the distances to all clusters

    >>> model.set_params(extra_applyout_settings={'mode': 'all_distances'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  DISTANCE_TO_CENTROID_1               ... DISTANCE_TO_CENTROID_5
    0   30                  3               ...      1.160697
    1   63                  4               ...      1.160697
    2   66                  3               ...      1.160697

    Saving the model in the schema named 'MODEL_STORAGE'
    Please model_storage class for further features of model storage.

    >>> model_storage = ModelStorage(connection_context=CONN, schema='MODEL_STORAGE')
    >>> model.name = 'My model name'
    >>> model_storage.save_model(model=model)

    Reloading the model for further use

    >>> model2 = AutoUnsupervisedClustering(conn_context=CONN)
    >>> model2.load_model(schema_name='MySchema', table_name='MyTable')
    >>> applyout2 = model2.predict(hana_df)
    >>> applyout2.head(3).collect()
        id  CLOSEST_CLUSTER_1  DISTANCE_TO_CLOSEST_CENTROID_1
    0   30                  3                        0.640378
    1   63                  4                        0.611050
    2   66                  3                        0.640378
    """

    def fit(self, data, key=None, features=None, weight=None):
        """
        Fits the model.

        Parameters
        ----------
        data : hana_ml DataFrame
            The training dataset
        key : str, optional
            The name of the ID column.
            This column will not be used as feature in the model.
            It will be output as row-id when prediction is made with the model.
            If `key` is not provided, an internal key is created. But this is not recommended.
        features : list of str, optional
            The names of the features to be used in the model.
            If `features` is not provided, all columns will be taken except the ID column.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        self : object
        """
        return self._fit(data=data,
                         key=key,
                         features=features,
                         label=None,
                         weight=weight)

    def _predict_clusters(self, data, mode, nb_distances):
        """
        Assigns clusters to data based on a fitted model.

        Parameters
        ----------
        data : hana_ml DataFrame
            The input dataset.
            It should have the same structure as the data used to fit the model.
        key : str
            The name of the ID column.
        features : list of str, optional.
            The names of the feature columns.
            If `features` is not provided, it defaults to all except the ID column.
        mode: str
            Defines the output to be generated when applying the model.
            There are two modes:
            - **closest_distances** : the top N closest clusters and distance to their centroids
                    will be output for each row
            - **all_distances** : the distance to each cluster will be output for each row
        nb_distances: int or str
            The number of distances to be generated in output from top clusters
            'all' means as many as there are clusters
        Returns
        -------
        hana_ml DataFrame
        """

        root_path = ('Protocols/Default/Transforms/Kxen.SmartSegmenter/Parameters/ApplySettings/' +
                     'UnSupervised/kxenUnsupervised/')
        apply_config_data_df = pd.DataFrame([
            ('APL/ApplyExtraMode', 'Advanced Apply Settings', None)])
        nb_distances = str(nb_distances)
        if nb_distances != 'all' and not nb_distances.isdigit():
            raise ValueError("The parameter 'nb_distances' must be either a integer or 'all'")
        if mode == 'closest_distances':
            # -- displays distance and cluster_ids ordered from closest to
            for val_type in ['PredictedRankNodeId', 'PredictedRankDistances']:
                param_path = (root_path + '{val_type}').format(val_type=val_type)
                if nb_distances == 'all':
                    apply_config_data_df = apply_config_data_df.append([
                        (param_path, 'all', None),
                        ])
                else:
                    apply_config_data_df = apply_config_data_df.append([
                        (param_path, 'individual', None)
                        ])
                    apply_config_data_df = apply_config_data_df.append([
                        ((param_path + '/' + str(nb_distances)), '', None),
                    ])
        elif mode == 'all_distances':
            # -- displays vector of the distances to clusters ordered by cluster_id
            val_type = 'PredictedNodeIdDistances'
            param_path = (root_path + '{val_type}').format(val_type=val_type)
            # Ignore nb_distances and consider it as 'all'
            apply_config_data_df = apply_config_data_df.append([
                (param_path, 'all', None),
                ])

        applyout_df = super(AutoUnsupervisedClustering, self)._predict(
            data=data, apply_config_data_df=apply_config_data_df)
        return self._rewrite_applyout_df(applyout_df=applyout_df, mode=mode)

    def fit_predict(self, data, key=None, features=None, weight=None):
        """
        Fits a clustering model and uses it to generate prediction output on the training dataset.

        Parameters
        ----------
        data : hana_ml DataFrame
            The input dataset
        key : str, optional
            The name of the ID column.
        features : list of str, optional.
            The names of the feature columns.
            If `features` is not provided, all non-ID columns will be taken.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        hana_ml DataFrame.
        The output is the same as the predict() method.

        Notes
        -----
        Please see the predict() method so as to get different outputs with the
        'extra_applyout_settings' parameter.
        """
        self.fit(data, key, features, weight=weight)
        return self.predict(data)

    def get_metrics(self):
        """
        Returns a dictionary containing the metrics about the model.

        Returns
        -------
        A dictionary object containing a set of clustering metrics and their values

        Examples
        --------

        >>> model.get_metrics()
        {'SimplifiedSilhouette': 0.14668968897882997,
         'RSS': 24462.640041325714,
         'IntraInertia': 3.2233573348587714,
         'KL': {1: OrderedDict([('hours-per-week', 0.2971627592049324),
                     ('occupation', 0.11944355994892383),
                     ('relationship', 0.06772624975990414),
                     ('education-num', 0.06377345492340795),
                     ('education', 0.06377345492340793),
                     ...}
        """
        cond = "VARIABLE='clusterId' and TARGET='clusterId' and DETAIL is null"
        dict1 = self._get_metrics(cond)
        # KLs
        cond = "VARIABLE='clusterId' and TARGET='clusterId' and DETAIL is not null"
        dict2 = self._get_metrics_by_cluster(cond=cond)
        # merge the two dictionaries
        dict1.update(dict2)
        return dict1

    def _rewrite_applyout_df(self, applyout_df, mode):
        """
        Rewrites the applyout dataframe so it outputs standardized column names.
        Arguments:
        ---------
        applyout_df : hana_ml DataFrame
            The initial output of predict

        Returns
        ------
        A new hana_ml DataFrame with renamed columns
        """
        # ---- maps columns
        map_new_cols = {}  # {new_column: old_column}
        key_col = None
        max_cluster_no = 0
        for i, old_col in enumerate(applyout_df.columns):
            cluster_id = None
            if i == 0:
                new_col = old_col  # ID column - no change
                key_col = new_col
            elif old_col == 'kc_clusterId':
                new_col = 'CLOSEST_CLUSTER_1'
                cluster_id = 1
            elif old_col == 'kc_best_dist_clusterId':
                new_col = 'DISTANCE_TO_CLOSEST_CENTROID_1'
                cluster_id = 1
            elif re.search(r'kc_best_dist_clusterId_(\d+)', old_col):
                cluster_id = re.search(r'kc_best_dist_clusterId_(\d+)', old_col).groups()[0]
                new_col = 'DISTANCE_TO_CLOSEST_CENTROID_' + cluster_id
            elif re.search(r'kc_clusterId_(\d+)', old_col):
                cluster_id = re.search(r'kc_clusterId_(\d+)', old_col).groups()[0]
                new_col = 'CLOSEST_CLUSTER_' + cluster_id
            elif re.search(r'kc_dist_cluster_clusterId_(\d+)', old_col):
                cluster_id = re.search(r'kc_dist_cluster_clusterId_(\d+)', old_col).groups()[0]
                new_col = 'DISTANCE_TO_CENTROID_' + cluster_id
            else:
                new_col = old_col
            if cluster_id:
                cluster_id = int(cluster_id)
                if cluster_id > max_cluster_no:
                    max_cluster_no = cluster_id
            # don't output CLOSEST_CLUSTER_1 (systematically provided by APL)
            # if mode=='all_distances'
            if (mode == 'all_distances') and (new_col == 'CLOSEST_CLUSTER_1'):
                continue
            if key_col != new_col:
                map_new_cols[new_col] = old_col
        return self._finalize_applyout_df(key_col, max_cluster_no, map_new_cols)


class AutoSupervisedClustering(_AutoClusteringBase):
    """
    SAP HANA APL Supervised Clustering algorithm.
    Clusters are determined with respect to a label variable.

    Parameters
    ----------
    label: str,
        The name of the label column
    nb_clusters : int, optional, default = 10
        The number of clusters to create
    nb_clusters_min: int, optional
        The minimum number of clusters to create.
        If the nb_clusters parameter is set, it will be ignored.
    nb_clusters_max: int, optional
        The maximum number of clusters to create.
        If the nb_clusters parameter is set, it will be ignored.
    distance: str, optional, default = 'SystemDetermined'
        The metric used to measure the distance between data points.
        The possible values are: 'L1', 'L2', 'LInf', 'SystemDetermined'.
    variable_storages: dict, optional
        Specifies the variable data types (string, integer, number).
        For example, {'VAR1': 'string', 'VAR2': 'number'}.
        See notes below for more details.
    variable_value_types: dict, optional
        Specifies the variable value types (continuous, nominal, ordinal).
        For example, {'VAR1': 'continuous', 'VAR2': 'nominal'}.
        See notes below for more details.
    variable_missing_strings: dict, optional
        Specifies the variable values that will be taken as missing.
        For example, {'VAR1': '???'} means anytime the variable value equals '???',
        it will be taken as missing.
    extra_applyout_settings: dict optional
        Defines the output to generate when applying the model.
        See documentation on predict() method for more information.
    other_params: dict optional
        Corresponds to the advanced settings.
        The dictionary contains {<parameter_name>: <parameter_value>}.
        The possible parameters are:
            - calculate_cross_statistics
            - calculate_sql_expressions
            - cutting_strategy
            - encoding_strategy
        See *Common APL Aliases for Model Training* in the `SAP HANA APL Reference Guide
        <https://help.sap.com/viewer/p/apl>`_.
    other_train_apl_aliases: dict, optional
        Users can provide APL aliases as advanced settings to the model.
        Unlike 'other_params' described above, users are free to input any possible value.
        There is no control in python.

    Attributes
    ----------
    model_ : hana_ml DataFrame
        The trained model content
    summary_ : APLArtifactTable
        The reference to the "SUMMARY" table generated by the model training.
        This table contains the summary about the model training.
    indicators_ : APLArtifactTable
        The reference to the "INDICATORS" table generated by the model training.
        This table contains the various metrics related to the model and its variables.
    fit_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table generated by the model training
    var_desc_ : APLArtifactTable
        The reference to the "VARIABLE_DESCRIPTION" table that was built during the model training
    applyout_ : hana_ml DataFrame
        The predictions generated the last time the model was applied
    predict_operation_logs_: APLArtifactTable
        The reference to the "OPERATION_LOG" table when a prediction was made

    Notes
    -----
        - The algorithm may detect less clusters than requested.
        This happens when a cluster detected on the estimation dataset was not found on
        the validation dataset. In that case, this cluster will be considered unstable and will
        then be removed from the model.
        Users can get the number of clusters actually found in the "INDICATORS" table. For example,
        ::
            # The actual number of clusters found
            d = model_u.get_indicators().collect()
            d[d.KEY=='FinalClusterCount'][['KEY','VALUE']]

        - It is highly recommended to use a dataset with a key provided in the fit() method.
        If not, once the model is trained, it will not be possible anymore to
        use the predict() method with a key, because the model will not expect it.

        - By default, when it is not given, SAP HANA APL guesses the variable description by reading
        the first 100 rows. But, sometimes, it does not provide the correct result.
        By specifically providing values for these parameters, the user can overwrite the default
        guess. For example:
        ::
            model.set_params(
                    variable_storages = {
                        'ID': 'integer',
                        'sepal length (cm)': 'number'
                        })
            model.set_params(
                    variable_value_types = {
                        'sepal length (cm)': 'continuous'
                        })
            model.set_params(
                    variable_missing_strings = {
                        'sepal length (cm)': '-1'
                        })

    Examples
    --------
    >>> from hana_ml.algorithms.apl.clustering import AutoSupervisedClustering
    >>> from hana_ml.dataframe import ConnectionContext, DataFrame

    Connecting to SAP HANA

    >>> CONN = ConnectionContext('HDB_HOST', HDB_PORT, 'HDB_USER', 'HDB_PASS')
    >>> # -- Creates Hana DataFrame
    >>> hana_df = DataFrame(CONN, 'select * from APL_SAMPLES.CENSUS')

    Creating and fitting the model

    >>> model = AutoSupervisedClustering(nb_clusters=5)
    >>> model.fit(data=hana_df, key='id', label='class')

    Debriefing

    >>> model.get_metrics()
    OrderedDict([('SimplifiedSilhouette', 0.3448029020802121), ('RSS', 4675.706587754118),...

    >>> model.get_metrics_by_cluster()
    {'Frequency': {1: 0.15139770759462357,
      2: 0.39707539649817214,
      3: 0.21549710013468568,
      4: 0.12949066820593166,
      5: 0.10653912756658696},
     'IntraInertia': {1: 0.1604412809425719,
      2: 0.10561882166246073,
      3: 0.12004212490063185,
      4: 0.21030892961293207,
      5: 0.08625667904000194},
     'RSS': {1: 883.710575431686,
      2: 1525.7694977359076,
      3: 941.1302592209537,
      4: 990.765367406523,
      5: 334.3308879590475},
     'SimplifiedSilhouette': {1: 0.3355726073943343,
      2: 0.4231738907945281,
      3: 0.2448648428415369,
      4: 0.38136325589137554,
      5: 0.22353657540054947},
     'TargetMean': {1: 0.1744734931009441,
      2: 0.022912917070469333,
      3: 0.3895408163265306,
      4: 0.7537677775419231,
      5: 0.21207430340557276},
     'TargetStandardDeviation': {1: 0.37951613049526484,
      2: 0.14962591788119842,
      3: 0.48764615116105525,
      4: 0.4308154072006165,
      5: 0.40877719266198526},
     'KL': {1: OrderedDict([('relationship', 0.6840012706191696),
                   ('education', 0.675109873839992),
                   ('education-num', 0.6751098738399919),
                   ('marital-status', 0.5806503390741476),
                   ('occupation', 0.46891689485806354),
                   ('sex', 0.08802303491483551),
                   ('capital-gain', 0.08794254258565125),
                   ...

    Predicting which cluster a data point belongs to

    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  CLOSEST_CLUSTER_1  DISTANCE_TO_CLOSEST_CENTROID_1
    0   30                  3                        0.640378
    1   63                  4                        0.611050
    2   66                  3                        0.640378
    3  110                  4                        0.611050
    4  335                  1                        0.851054

    Determining the 2 closest clusters

    >>> model.set_params(extra_applyout_settings={'mode':'closest_distances', 'nb_distances': 2})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  CLOSEST_CLUSTER_1  ...  CLOSEST_CLUSTER_2  DISTANCE_TO_CLOSEST_CENTROID_2
    0   30                  3  ...                  4                        0.730330
    1   63                  4  ...                  1                        0.851054
    2   66                  3  ...                  4                        0.730330
    3  110                  4  ...                  1                        0.851054
    4  335                  1  ...                  4                        0.906003

    Retrieving the distances to all clusters

    >>> model.set_params(extra_applyout_settings={'mode': 'all_distances'})
    >>> applyout_df = model.predict(hana_df)
    >>> applyout_df.collect() # returns the output as a pandas DataFrame
        id  DISTANCE_TO_CENTROID_1               ... DISTANCE_TO_CENTROID_5
    0   30                0.851054               ...      1.160697
    1   63                0.751054               ...      1.160697
    2   66                0.906003               ...      1.160697

    Saving the model in the schema named 'MODEL_STORAGE'
    Please see model_storage class for further features of model storage.

    >>> model_storage = ModelStorage(connection_context=CONN, schema='MODEL_STORAGE')
    >>> model.name = 'My model name'
    >>> model_storage.save_model(model=model, if_exists='replace')

    Reloading the model for further uses
    Please note that the label has to be specified again prior to calling predict()

    >>> model2 = AutoSupervisedClustering()
    >>> model2.set_params(label='class')
    >>> model2.load_model(schema_name='MySchema', table_name='MyTable')
    >>> applyout2 = model2.predict(hana_df)
    >>> applyout2.head(3).collect()
        id  CLOSEST_CLUSTER_1  DISTANCE_TO_CLOSEST_CENTROID_1
    0   30                  3                        0.640378
    1   63                  4                        0.611050
    2   66                  3                        0.640378
    """
    def __init__(self,
                 conn_context=None,
                 label=None,
                 nb_clusters=None,
                 nb_clusters_min=None,
                 nb_clusters_max=None,
                 distance=None,
                 variable_storages=None,
                 variable_value_types=None,
                 variable_missing_strings=None,
                 extra_applyout_settings=None,
                 ** other_params): #pylint: disable=too-many-arguments
        super(AutoSupervisedClustering, self).__init__(
            conn_context=conn_context,
            nb_clusters=nb_clusters,
            nb_clusters_min=nb_clusters_min,
            nb_clusters_max=nb_clusters_max,
            distance=distance,
            variable_storages=variable_storages,
            variable_value_types=variable_value_types,
            variable_missing_strings=variable_missing_strings,
            extra_applyout_settings=extra_applyout_settings,
            ** other_params)
        self.label = self._arg('label', label, str)

    def set_params(self, **parameters):
        """
        Sets attributes of the current model

        Parameters
        ----------
        params : dictionary
            containing attribute names and values
        """
        if 'label' in parameters:
            self.label = self._arg('label', parameters.pop('label'), str)
        if parameters:
            super(AutoSupervisedClustering, self).set_params(**parameters)
        return self

    # pylint: disable=too-many-arguments
    def fit(self, data, key=None, label=None, features=None, weight=None):
        """
        Fits the model.

        Parameters
        ----------
        data : hana_ml DataFrame
            The training dataset
        key : str, optional
            The name of the ID column.
            This column will not be used as feature in the model.
            It will be output as row-id when prediction is made with the model.
            If `key` is not provided, an internal key is created. But this is not recommended.
        label : str, option
            The name of the label column.
            If it is not given, the model 'label' attribute will be taken.
            If this latter is not defined, an error will be raised.
        features : list of str, optional
            The names of the features to be used in the model.
            If `features` is not provided, all columns will be taken except the ID and the label
            columns.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        self : object
        """
        if label:
            self.label = self._arg('label', label, str, required=True)
        elif not self.label:
            raise TypeError('label is required')
        return self._fit(data=data,
                         key=key,
                         features=features,
                         label=self.label,
                         weight=weight)

    def predict(self, data):
        """
        Predicts which cluster each specified row belongs to.

        Parameters
        ----------
        data : hana_ml DataFrame
            The set of rows for which to generate cluster predictions.
            This dataset must have the same structure as the one used in the fit() method.

        Returns
        -------
        hana_ml DataFrame
            By default, the ID of the closest cluster and the distance to its center are provided.
            Users can request different outputs by setting the **extra_applyout_settings** parameter
            in the model.
            The **extra_applyout_settings** parameter is a dictionary with **'mode'** and
            **'nb_distances'** as keys.
            If **mode** is set to **'closest_distances'**, cluster IDs and distances to centroids
            will be provided from the closest to the furthest cluster.
            The output columns will be:
                - <The key column name>,
                - CLOSEST_CLUSTER_1,
                - DISTANCE_TO_CLOSEST_CENTROID_1,
                - CLOSEST_CLUSTER_2,
                - DISTANCE_TO_CLOSEST_CENTROID_2,
                ...
            If **mode** is set to **'all_distances'**, the distances to the centroids of all
            clusters will be provided in cluster ID order. The output columns will be:
                - ID,
                - DISTANCE_TO_CENTROID_1,
                - DISTANCE_TO_CENTROID_2,
                ...
            **nb_distances** limits the output to the closest clusters. It is only valid when
            **mode** is **'closest_distances'** (it will be ignored if **mode** = 'all distances').
            It can be set to **'all'** or a positive integer.

        Examples
        --------

        Retrieves the IDs of the 3 closest clusters and the distances to their centroids:

        >>> extra_applyout_settings = {'mode': 'closest_distances', 'nb_distances': 3}
        >>> model.set_params(extra_applyout_settings=extra_applyout_settings)
        >>> out = model.predict(hana_df)
        >>> out.head(3).collect()
            id  CLOSEST_CLUSTER_1  ...  CLOSEST_CLUSTER_3  DISTANCE_TO_CLOSEST_CENTROID_3
        0   30                  3  ...                  4                        0.730330
        1   63                  4  ...                  1                        0.851054
        2   66                  3  ...                  4                        0.730330

        Retrieves the distances to all clusters:

        >>> model.set_params(extra_applyout_settings={'mode': 'all_distances'})
        >>> out = model.predict(hana_df)
        >>> out.head(3).collect()
           id  DISTANCE_TO_CENTROID_1  DISTANCE_TO_CENTROID_2  ... DISTANCE_TO_CENTROID_5
        0  30                0.994595                0.877414  ...              0.782949
        1  63                0.994595                0.985202  ...              0.782949
        2  66                0.994595                0.877414  ...              0.782949
        """
        if not self.label:
            # The label for which the model was trained is required.
            # It is used when we specify the model parameter for apply:
            # Protocols/Default/Transforms/Kxen.SmartSegmenter/Parameters/ApplySettings/
            # Supervised/{label}/
            raise Error('Cannot make predictions. Label parameter must be defined so that the'
                        ' output of predict() can be correctly parameterized')
        return super(AutoSupervisedClustering, self).predict(data)

    def _predict_clusters(self, data, mode, nb_distances):
        """
        Assigns clusters to data based on a fitted model.

        Parameters
        ----------
        data : DataFrame
            The input dataset.
            This dataframe's column structure should match that of the data used for fit().
        mode: str
            Specifies the way the distances are to be provided in output.
            Two values are possible:
            - 'closest_distances' : provides the distances to the clusters in the ascending order of
                                 the distance values (from closest to farthest)
            - 'all_distances' : provides the distances to the clusters in ascending order of the
                                cluster id (cluster 1 to N)
        nb_distances: str ('all') or int
            The number of distances to be generated in output from top clusters
            'all' means as many as there are clusters
        Returns
        -------
        hana_ml DataFrame
        The columns of the dataframe depends on the input parameters 'mode' and 'nb_distances'.
        In all cases, the two first columns are always:
              - <ID> : Data point ID, with name and type taken from the input ID column.
              - CLOSEST_CLUSTER_1 : INTEGER type, the closest cluster ID.
        If 'mode' is 'closest_distances', the next columns will be the distances and the cluster ids
        ordered by distance values (from closest to farthest).
        If 'mode' is 'all_distances', the next columns will be the distances to the clusters
        ordered by cluster ID.
        """
        root_path = ('Protocols/Default/Transforms/Kxen.SmartSegmenter/Parameters/ApplySettings/'
                     + 'Supervised/{label}/').format(label=self.label)
        apply_config_data_df = pd.DataFrame([
            ('APL/ApplyExtraMode', 'Advanced Apply Settings', None)])
        nb_distances = str(nb_distances)
        if nb_distances != 'all' and not nb_distances.isdigit():
            raise ValueError("The parameter 'nb_distances' must be either a integer or 'all'")
        if mode == 'closest_distances':
            # -- displays distance and cluster_ids ordered from closest to farthest
            for val_type in ['PredictedRankNodeId', 'PredictedRankDistances']:
                param_path = (root_path + '{val_type}').format(val_type=val_type)
                if nb_distances == 'all':
                    apply_config_data_df = apply_config_data_df.append([
                        (param_path, 'all', None),
                        ])
                else:
                    apply_config_data_df = apply_config_data_df.append([
                        (param_path, 'individual', None)
                        ])
                    apply_config_data_df = apply_config_data_df.append([
                        ((param_path + '/' + str(nb_distances)), '', None),
                    ])
        elif mode == 'all_distances':
            # -- displays vector of the distances to clusters ordered by cluster_id
            val_type = 'PredictedNodeIdDistances'
            param_path = (root_path + '{val_type}').format(val_type=val_type)
            # Ignore nb_distances and consider it as 'all'
            apply_config_data_df = apply_config_data_df.append([
                (param_path, 'all', None),
                ])
        applyout_df = super(AutoSupervisedClustering, self)._predict(
            data=data, apply_config_data_df=apply_config_data_df)
        return self._rewrite_applyout_df(applyout_df=applyout_df, mode=mode)

    # pylint: disable=too-many-arguments
    def fit_predict(self, data, key=None, label=None, features=None, weight=None):
        """
        Fits a clustering model and uses it to generate prediction output on the training dataset.

        Parameters
        ----------
        data : hana_ml DataFrame
            The input dataset
        key : str, optional
            The name of the ID column
        label : str
            The name of the label column
        features : list of str, optional.
            The names of the feature columns.
            If `features` is not provided, all non-ID and non-label columns will be taken.
        weight : str, optional
            The name of the weight variable.
            A weight variable allows one to assign a relative weight to each of the observations.

        Returns
        -------
        hana_ml DataFrame.
        The output is the same as the predict() method.

        Notes
        -----
        Please see the predict() method so as to get different outputs with the
        'extra_applyout_settings' parameter.
        """
        self.fit(data=data, key=key, label=label, features=features, weight=weight)
        return self.predict(data)

    def get_metrics(self):
        """
        Returns a dictionary containing the metrics about the model.

        Returns
        -------
        A dictionary object containing a set of clustering metrics and their values

        Examples
        --------

        >>> model.get_metrics()
        {'SimplifiedSilhouette': 0.14668968897882997,
         'RSS': 24462.640041325714,
         'IntraInertia': 3.2233573348587714,
         'Frequency': {
            1: 0.3167862345729914,
            2: 0.35590005772243755,
            3: 0.3273137077045711},
         'IntraInertia': {1: 0.7450335510518645,
             2: 0.708350629565789,
             3: 0.7006679558645009},
         'RSS': {1: 8586.511675872738,
             2: 9171.723951617836,
             3: 8343.554018434477},
         'SimplifiedSilhouette': {1: 0.13324659043317924,
             2: 0.14182734764281074,
             3: 0.1311620470933516},
         'TargetMean': {1: 0.1744734931009441,
              2: 0.022912917070469333,
              3: 0.3895408163265306},
         'TargetStandardDeviation': {1: 0.37951613049526484,
              2: 0.14962591788119842,
              3: 0.48764615116105525},
         'KL': {1: OrderedDict([('hours-per-week', 0.2971627592049324),
                     ('occupation', 0.11944355994892383),
                     ('relationship', 0.06772624975990414),
                     ('education-num', 0.06377345492340795),
                     ('education', 0.06377345492340793),
                     ...
        """
        cond1 = "VARIABLE='{label}' and TARGET='{label}' and DETAIL is null".format(
            label=self.label)
        cond = '({cond1}) or ({cond2})'.format(
            cond1=cond1,
            cond2="TO_VARCHAR(DETAIL) like 'kc_%'"  # for KI, KR, AUC, ...
            )
        dict1 = self._get_metrics(cond)

        # Get KLs
        cond = ("VARIABLE='{label}' and TARGET='{label}'"
                + " and DETAIL is not null"
                + " and to_varchar(DETAIL) not like 'kc%'").format(label=self.label)
        dict2 = self._get_metrics_by_cluster(cond=cond, label=self.label)
        dict1.update(dict2)
        return dict1

    def load_model(self, schema_name, table_name, oid=None):
        """
        Loads the model from a table.

        Parameters
        ----------
        schema_name: str
            The schema name
        table_name: str
            The table name
        oid : str. optional
            If the table contains several models,
            the OID must be given as an identifier.
            If it is not provided, the whole table is read.
        Notice
        ------
        Prior to using a reloaded model for a new prediction, it is necessary to re-specify
        the 'label' parameter. Otherwise, the predict() method will fail.

        Example
        -------

        >>> # needs to re-specify time_column_name for view creation
        >>> model = AutoTimeSeries(label='class')
        >>> model.load_model(schema_name='MY_SCHEMA',
        >>>          table_name='MY_MODEL_TABLE',
        >>>          )
        >>> model.predict(hana_df)

        """
        super(AutoSupervisedClustering, self).load_model(
            schema_name=schema_name,
            table_name=table_name,
            oid=oid)
        if not self.label:
            logger.warning("The label parameter is not defined. "
                           "Set it to a correct value before calling predict().")

    def _rewrite_applyout_df(self, applyout_df, mode): #pylint: disable=too-many-branches
        """
        Rewrites the applyout dataframe so it outputs standardized column names.
        Parameters:
        ---------
        applyout_df : hana ml DataFrame
            The initial output of predict

        Returns
        ------
        A new hana_ml DataFrame with renamed columns
        """

        # ---- maps columns
        # ---- maps columns
        map_new_cols = {}  # {new_column: old_column}
        key_col = None
        max_cluster_no = 0
        for i, old_col in enumerate(applyout_df.columns):
            cluster_id = None
            if i == 0:
                new_col = old_col  # ID column - no change
                key_col = new_col
            elif old_col == ('kc_' + self.label):
                new_col = 'CLOSEST_CLUSTER_1'
                cluster_id = 1
            elif old_col == ('kc_best_dist_' + self.label):
                new_col = 'DISTANCE_TO_CLOSEST_CENTROID_1'
                cluster_id = 1
            elif re.search(r'kc_{label}_(\d+)'.format(label=self.label), old_col):
                cluster_id = re.search(r'kc_{label}_(\d+)'.format(label=self.label),
                                       old_col).groups()[0]
                new_col = 'CLOSEST_CLUSTER_' + cluster_id
            elif re.search(r'kc_best_dist_{label}_(\d+)'.format(label=self.label), old_col):
                cluster_id = re.search(r'kc_best_dist_{label}_(\d+)'.format(label=self.label),
                                       old_col).groups()[0]
                new_col = 'DISTANCE_TO_CLOSEST_CENTROID_' + cluster_id
            elif re.search(r'kc_dist_cluster_{label}_(\d+)'.format(label=self.label), old_col):
                cluster_id = re.search(r'kc_dist_cluster_{label}_(\d+)'.format(label=self.label),
                                       old_col).groups()[0]
                new_col = 'DISTANCE_TO_CENTROID_' + cluster_id
            elif old_col == self.label:
                continue
            else:
                new_col = old_col
            if cluster_id:
                cluster_id = int(cluster_id)
                if cluster_id > max_cluster_no:
                    max_cluster_no = cluster_id
            # don't output CLOSEST_CLUSTER_1 (systematically provided by APL)
            # if mode=='all_distances'
            if (mode == 'all_distances') and (new_col == 'CLOSEST_CLUSTER_1'):
                continue
            if key_col != new_col:
                map_new_cols[new_col] = old_col
        return self._finalize_applyout_df(key_col, max_cluster_no, map_new_cols)
