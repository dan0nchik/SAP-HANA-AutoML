"""
This module contain Python wrapper for the PAL partition function.

The following function is available:

    * :func:`train_test_val_split`
"""

import logging
import uuid
from hdbcli import dbapi #pylint: disable=import-error
from hana_ml import dataframe
from hana_ml.dataframe import quotename
from .pal_base import (
    ParameterTable,
    arg,
    try_drop,
    materialize,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__) #pylint: disable=invalid-name
#pylint:disable=too-many-statements, too-many-branches, too-many-locals, too-few-public-methods, too-many-arguments
def train_test_val_split(data, id_column=None, random_seed=None, thread_ratio=None,
                         partition_method='random', stratified_column=None,
                         training_percentage=None, testing_percentage=None,
                         validation_percentage=None, training_size=None,
                         testing_size=None, validation_size=None):

    """
    The algorithm partitions an input dataset randomly into three disjoint
    subsets called training, testing and validation.
    Let us remark that the union of these three subsets might not be the
    complete initial dataset.

    Please also note that the dataset must have an ID column. The ID column can
    be specified explicitly, otherwise it's assumed that the first column of the
    dataframe holds the ID.

    Two different partitions can be obtained:

    1. Random Partition, which randomly divides all the data.
    2. Stratified Partition, which divides each subpopulation randomly.

    In the second case, the dataset needs to have at least one categorical
    attribute (for example, of type VARCHAR). The initial dataset will first be
    subdivided according to the different categorical values of this attribute.
    Each mutually exclusive subset will then be randomly split to obtain the
    training, testing, and validation subsets.This ensures that all
    "categorical values" or "strata" will be present in the sampled subset.

    Parameters
    ----------

    data : DataFrame
        DataFrame to be partitioned.

    id_column: str, optional
        Indicates which column to use as the ID column,
        Defauls to first column.

    random_seed : int, optional
        Indicates the seed used to initialize the random number generator.
            - 0: Uses the system time.
            - Not 0: Uses the specified seed.

        Defaults to 0.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.

        Values outside the range will be ignored and this function heuristically
        determines the number of threads to use.

        Defaults to 0.

    partition_method : {'random', 'stratified'}, optional
        Partition method:
            - 'random': random partitions.
            - 'stratified': stratified partition.

        Defaults to 'random'.

    stratified_column : str, optional
        Indicates which column is used for stratification.

        Valid only when ``parition_method`` is set to 'stratified' (stratified partition).

        No default value.

    training_percentage : float, optional
        The percentage of training data.

        Value range: 0 <= value <= 1.

        Defaults to 0.8.

    testing_percentage : float, optional
        The percentage of testing data.

        Value range: 0 <= value <= 1.

        Defaults to 0.1.

    validation_percentage : float, optional
        The percentage of validation data.

        Value range: 0 <= value <= 1.

        Defaults to 0.1.

    training_size : int, optional
        Row size of training data. Value range: >=0.

        If both ``training_percentage`` and ``training_size`` are specified,
        ``training_percentage`` takes precedence.

        No default value.

    testing_size : int, optional
        Row size of testing data. Value range: >=0.

        If both ``testing_percentage`` and ``testing_size`` are specified,
        ``testing_percentage`` takes precedence.

        No default value.

    validation_size : int, optional
        Row size of validation data. Value range:>=0.

        If both ``validation_percentage`` and ``validation_size`` are specified,
        ``validation_percentage`` takes precedence.

        No default value.

    Returns
    -------

    Returns three DataFrame of training data, testing data and validation data after partition.

    Examples
    --------
    To partition the input DataFrame df:

    >>> train, test, valid = train_test_val_split(data=df)
    """

    # SQLTRACE
    conn = data.connection_context
    require_pal_usable(conn)
    conn.sql_tracer.set_sql_trace(None, 'partition', 'train_test_val_split')

    thread_ratio = arg('thread_ratio', thread_ratio, float)

    partition_method_map = {'random': 0, 'stratified': 1}
    partition_method = arg('partition_method', partition_method,
                           partition_method_map)

    training_percentage = arg('training_percentage', training_percentage, float)
    testing_percentage = arg('testing_percentage', testing_percentage, float)
    validation_percentage = arg('validation_percentage', validation_percentage, float)

    training_size = arg('training_size', training_size, int)
    testing_size = arg('testing_size', testing_size, int)
    validation_size = arg('validation_size', validation_size, int)

    stratified_column = arg('stratified_column', stratified_column, str)
    if training_percentage is not None and testing_percentage is not None\
        and validation_percentage is not None:
        if training_percentage + testing_percentage + validation_percentage != 1.0:
            ## error message
            msg = ("the sum of training_percentage, testing_percentage and " +\
                   "validation_percentage must be equal to 1")
            logger.error(msg)
            raise ValueError(msg)

    if stratified_column is not None and partition_method == 0:
        ## error message
        msg = ("stratified column needs to be specified when PARTITION_METHOD " +
               "is set to stratified")
        logger.error(msg)
        raise ValueError(msg)

    # determine ID column: if specified move to first column, otherwise it's assumed
    # that it's the first column
    if id_column is not None:
        id_column_name = id_column
        data = data.to_head(id_column_name)
    else:
        id_column_name = data.dtypes()[0][0]

    # get column name and data type of input data first column
    id_column_name = data.dtypes()[0][0]
    if data.dtypes()[0][1] == 'INT' or data.dtypes()[0][1] == 'BIGINT':   # pylint: disable=fixme
        id_column_type = 'INTEGER'#pylint:disable=unused-variable
    else:  # column type is varchar or nvarchar
        id_column_type = data.dtypes()[0][1] + '(' + str(data.dtypes()[0][2]) + ')'#pylint:disable=unused-variable

    # check for BIGINT columns other than ID
    # create dictionary with the colummn name/ data type as key/values for all columns
    # except first one (which is the ID column where BIGINT is supported)
    columns_dtypes_dict = {i[0]:i[1] for i in data.dtypes()[1:]}

    # drop unsupported data types (BIGINT) from dataframe
    # column(s) will be added back during final join operation
    data = data.drop([d for d in columns_dtypes_dict.keys() if columns_dtypes_dict[d] == 'BIGINT'])

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    data_tbl = "#PAL_PARTITION_DATA_TBL_{}".format(unique_id)
    materialize(conn, data_tbl, data)
    result_tbl = '#PAL_PARTITION_RESULT_TBL_{}'.format(unique_id)

    # param table manipulation
    param_rows = [('RANDOM_SEED', random_seed, None, None),
                  ('THREAD_RATIO', None, thread_ratio, None),
                  ('PARTITION_METHOD', partition_method, None, None),
                  ('TRAINING_PERCENT', None, training_percentage, None),
                  ('TESTING_PERCENT', None, testing_percentage, None),
                  ('VALIDATION_PERCENT', None, validation_percentage, None),
                  ('TRAINING_SIZE', training_size, None, None),
                  ('TESTING_SIZE', testing_size, None, None),
                  ('VALIDATION_SIZE', validation_size, None, None)]
    if partition_method != 0:  # doing stratified paritioning, need column
        param_rows.extend([('STRATIFIED_COLUMN', None, None, stratified_column)])

    try:
        call_pal_auto(conn,
                      'PAL_PARTITION',
                      data,
                      ParameterTable().with_data(param_rows),
                      result_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise

    train_sql = 'SELECT a.* FROM {} a inner join {} b \
       on a.{} = b.{} where b."PARTITION_TYPE" = 1' .format(data_tbl, result_tbl,
                                                            quotename(id_column_name),
                                                            quotename(id_column_name))
    train_ = dataframe.DataFrame(conn, train_sql)#pylint:disable=attribute-defined-outside-init
     # SQLTRACE
    conn.sql_tracer.trace_object({
        'name': 'train',
        'table_type': train_.generate_table_type(),
        'select': train_.select_statement
    }, sub_cat='output_tables')

    test_sql = 'SELECT a.* FROM {} a inner join {} b \
       on a.{} = b.{} where b."PARTITION_TYPE" = 2' .format(data_tbl, result_tbl,
                                                            quotename(id_column_name),
                                                            quotename(id_column_name))
    test_ = dataframe.DataFrame(conn, test_sql)#pylint:disable=attribute-defined-outside-init
     # SQLTRACE
    conn.sql_tracer.trace_object({
        'name': 'test',
        'table_type': test_.generate_table_type(),
        'select': test_.select_statement
    }, sub_cat='output_tables')

    validation_sql = 'SELECT a.* FROM {} a inner join {} b \
       on a.{} = b.{} where b."PARTITION_TYPE" = 3' .format(data_tbl, result_tbl,
                                                            quotename(id_column_name),
                                                            quotename(id_column_name))
    validation_ = dataframe.DataFrame(conn, validation_sql)#pylint:disable=attribute-defined-outside-init
     # SQLTRACE
    conn.sql_tracer.trace_object({
        'name': 'validation',
        'table_type': validation_.generate_table_type(),
        'select': validation_.select_statement
    }, sub_cat='output_tables')

    return train_, test_, validation_
