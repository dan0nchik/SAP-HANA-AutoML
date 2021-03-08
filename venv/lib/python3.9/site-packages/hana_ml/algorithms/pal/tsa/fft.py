"""
This module contains Python wrapper for PAL Fast-Fourier-Transform(FFT) algorithm.

The following class is available:

    * :class:`FFT`
    * :func:`fft`
"""

# pylint: disable=line-too-long, too-few-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
import logging
import uuid
import pandas as pd
from hdbcli import dbapi
from hana_ml.dataframe import create_dataframe_from_pandas
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    #Table,
    ParameterTable,
    #INTEGER,
    #DOUBLE,
    try_drop,
    require_pal_usable,
    call_pal_auto
)
logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class FFT(PALBase):
    r"""
    Fast Fourier Transform is applied to discrete data sequence.

    Parameters
    ----------

    None

    Attributes
    ----------

    None

    Examples
    --------

    Training data df:

    >>> df.collect()
       ID  RE   IM
    0   1  2.0  9.0
    1   2  3.0 -3.0
    2   3  5.0  0.0
    3   4  0.0  0.0
    4   5 -2.0 -2.0
    5   6 -9.0 -7.0
    6   7  7.0  0.0

    Create a FFT instance:

    >>> fft = FFT()

    Call apply() on given data sequence:

    >>> result = fft.apply(data=df, inverse=False)
    >>> result.collect()
       ID   REAL        IMAG
    0   1   6.000000   -3.000000
    1   2   16.273688  -0.900317
    2   3  -5.393946    26.265112
    3   4  -13.883222   18.514840
    4   5  -4.233990   -2.947800
    5   6   9.657319    3.189618
    6   7   5.580151    21.878547

    """

    number_type_map = {'real':1, 'imag':2}
    def apply(self, data, num_type=None, inverse=None, window=None,
              window_start=None, window_length=None, alpha=None,
              beta=None, attenuation=None, flattop_model=None, flattop_precision=None,
              r=None):
        """
        Apply Fast-Fourier-Transfrom to the input data, and return the transformed data.

        Parameters
        ----------
        data : DataFrame
          The DataFrame contains at most 3 columns, where:

            - The first column is ID, which indicates order of sequence.
            - The second column is the real part of the sequence.
            - The third column indicates imaginary part of the sequence which is optional.

        num_type : {'real', 'imag'}, optional
            Number type for the second column of the input data.
            Valid only when the input data contains 3 columns.

            Defaults to 'real'.
        inverse : bool, optional
            If False, forward FFT is applied; otherwise inverse FFT is applied.

            Defaults to False.
        window : str, optional
            Available input

            - 'none'
            - 'hamming'
            - 'hann'
            - 'hanning'
            - 'bartlett'
            - 'triangular'
            - 'bartlett_hann'
            - 'blackman'
            - 'blackman_harris'
            - 'blackman_nuttall'
            - 'bohman'
            - 'cauchy'
            - 'cheb'
            - 'chebwin'
            - 'cosine'
            - 'sine'
            - 'flattop'
            - 'gaussian'
            - 'kaiser'
            - 'lanczos'
            - 'sinc'
            - 'nuttall'
            - 'parzen'
            - 'poisson'
            - 'poisson_hann'
            - 'poisson_hanning'
            - 'rectangle'
            - 'riemann'
            - 'riesz'
            - 'tukey'

            Only available for pure real forward FFT.
        window_start : int, optional
            Specifies the starting point of tapering window.

            Defaults to 0.
        window_length : int, optional
            Specifies the length of Tapering Window.
        alpha : float, optional
            Parameter for the Window below:

            - Blackman, defaults to 0.16
            - Cauchy, defaults to 3.0
            - Gaussian, defaults to 2.5
            - Poisson, defaults to 2.0
            - Poisson_hann(Poisson_hanning), defaults to 2.0

        beta : float, optional
            Parameter for Kaiser Window.

            Defaults to 8.6.
        attenuation : float, optional
            Parameter for Cheb(Chebwin).

            Defaults to 50.0.
        flattop_model : str, optional
            Parameter for Flattop Window. Can be:

              - 'symmetric'
              - 'periodic'

            Defaults to 'symmetric'.
        flattop_precision : str, optional
            Parameter for Flattop Window. Can be:

              - 'none'
              - 'octave'

            Defulats to 'none'.
        r : float, optional
            Parameter for Tukey Window.

            Defualts to 0.5.

        Returns
        -------
        DataFrame
            Dataframe containing the transformed sequence, structured as follows:
                - 1st column: ID, with same name and type as input data.
                - 2nd column: REAL, type DOUBLE, representing real part of the transformed sequence.
                - 3rd column: IMAG, type DOUBLE, represneting imaginary part of the transformed sequence.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        inverse = self._arg('inverse', inverse, bool)
        num_type = self._arg('num_type', num_type, self.number_type_map)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        #tables = ['DATA', 'PARAM', 'RESULT']
        result_tbl = '#PAL_FFT_RESULT_TBL_{}_{}'.format(self.id, unique_id)
        #data_tbl, param_tbl, result_tbl = tables

        param_rows = [
            ('INVERSE', None if inverse is None else int(inverse), None, None),
            ('NUMBER_TYPE',
             None if num_type is None else int(num_type),
             None, None),
            ]
        if window is not None:
            param_rows.append(('WINDOW', None, None, window))
        if window_start is not None:
            param_rows.append(('WINDOW_START', window_start, None, None))
        if window_length is not None:
            param_rows.append(('WINDOW_LENGTH', window_length, None, None))
        if alpha is not None:
            param_rows.append(('ALPHA', None, alpha, None))
        if beta is not None:
            param_rows.append(('BETA', None, beta, None))
        if attenuation is not None:
            param_rows.append(('ATTENUATION', None, attenuation, None))
        if r is not None:
            param_rows.append(('R', None, attenuation, None))
        if flattop_model is not None:
            param_rows.append(('MODE', None, None, flattop_model))
        if flattop_precision is not None:
            param_rows.append(('PRECISION', None, None, flattop_precision))
        #result_spec = [
        #    ("ID", INTEGER),
        #    ("REAL", DOUBLE),
        #    ("IMAG", DOUBLE)
        #    ]
        try:
            #self._materialize(data_tbl, data)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))

            call_pal_auto(conn,
                          "PAL_FFT",
                          data,
                          ParameterTable().with_data(param_rows),
                          result_tbl)
        except dbapi.Error as db_err:
            #msg = ('HANA error while attempting to apply FFT.')
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise

        return conn.table(result_tbl)

def fft(data, num_type=None, inverse=None, window=None,
        window_start=None, window_length=None, alpha=None,
        beta=None, attenuation=None, flattop_model=None, flattop_precision=None,
        r=None):
    """
    Apply Fast-Fourier-Transfrom to the input data, and return the transformed data.

    Parameters
    ----------
    data : DataFrame
        The DataFrame contains at most 3 columns, where:

          - The first column is ID, which indicates order of sequence.
          - The second column is the real part of the sequence.
          - The third column indicates imaginary part of the sequence which is optional.

    num_type : {'real', 'imag'}, optional
        Number type for the second column of the input data.

        Valid only when the input data contains 3 columns.

        Defaults to 'real'.
    inverse : bool, optional
        If False, forward FFT is applied; otherwise inverse FFT is applied.

        Defaults to False.
    window : str, optional
        Available input

        - 'none'
        - 'hamming'
        - 'hann'
        - 'hanning'
        - 'bartlett'
        - 'triangular'
        - 'bartlett_hann'
        - 'blackman'
        - 'blackman_harris'
        - 'blackman_nuttall'
        - 'bohman'
        - 'cauchy'
        - 'cheb'
        - 'chebwin'
        - 'cosine'
        - 'sine'
        - 'flattop'
        - 'gaussian'
        - 'kaiser'
        - 'lanczos'
        - 'sinc'
        - 'nuttall'
        - 'parzen'
        - 'poisson'
        - 'poisson_hann'
        - 'poisson_hanning'
        - 'rectangle'
        - 'riemann'
        - 'riesz'
        - 'tukey'
        Only available for pure real forward FFT.
    window_start : int, optional
        Specifies the starting point of tapering window.

        Defaults to 0.
    window_length : int, optional
        Specifies the length of Tapering Window.
    alpha : float, optional
        Parameter for the Window below:

        - Blackman, defaults to 0.16
        - Cauchy, defaults to 3.0
        - Gaussian, defaults to 2.5
        - Poisson, defaults to 2.0
        - Poisson_hann(Poisson_hanning), defaults to 2.0
    beta : float, optional
        Parameter for Kaiser Window.

        Defaults to 8.6.
    attenuation : float, optional
        Parameter for Cheb(Chebwin).

        Defaults to 50.0.
    flattop_model : str, optional
        Parameter for Flattop Window. Can be:

          - 'symmetric'
          - 'periodic'

        Defaults to 'symmetric'.
    flattop_precision : str, optional
        Parameter for Flattop Window. Can be:

          - 'none'
          - 'octave'

        Defulats to 'none'.
    r : float, optional
        Parameter for Tukey Window.

        Defualts to 0.5.

    Returns
    -------

    DataFrame
        Dataframe containing the transformed sequence, structured as follows:

            - 1st column: ID, with same name and type as input data.
            - 2nd column: REAL, type DOUBLE, representing real part of the transformed sequence.
            - 3rd column: IMAG, type DOUBLE, represneting imaginary part of the transformed sequence.

    Examples
    --------

    Training data df:

    >>> df.collect()
       ID  RE   IM
    0   1  2.0  9.0
    1   2  3.0 -3.0
    2   3  5.0  0.0
    3   4  0.0  0.0
    4   5 -2.0 -2.0
    5   6 -9.0 -7.0
    6   7  7.0  0.0

    >>> result = fft(data=df, inverse=False)
    >>> result.collect()
       ID   REAL        IMAG
    0   1   6.000000   -3.000000
    1   2   16.273688  -0.900317
    2   3  -5.393946    26.265112
    3   4  -13.883222   18.514840
    4   5  -4.233990   -2.947800
    5   6   9.657319    3.189618
    6   7   5.580151    21.878547
    """
    fft_instance = FFT()
    return fft_instance.apply(data,
                              num_type,
                              inverse,
                              window,
                              window_start,
                              window_length,
                              alpha,
                              beta,
                              attenuation,
                              flattop_model,
                              flattop_precision,
                              r)

def massive_fft(data, num_type=None, inverse=None, window=None,
                window_start=None, window_length=None, alpha=None,
                beta=None, attenuation=None, flattop_model=None, flattop_precision=None,
                r=None):
    """
    Massive FFT computation. (internal use)
    """
    conn = data.connection_context
    schema = conn.get_current_schema()
    gen_id = str(uuid.uuid4()).upper().replace("-", "_")
    pal_massive_fft_data_tbl = "PAL_MASSIVE_FFT_DATA_TBL_" + gen_id
    pal_massive_fft_param_tbl = "PAL_MASSIVE_FFT_PARAM_TBL_" + gen_id
    pal_massive_fft_transformed_tbl = "PAL_MASSIVE_FFT_TRANSFORMED_TBL" + gen_id
    pal_massive_fft_error_tbl = "PAL_MASSIVE_FFT_ERROR_TBL" + gen_id
    pal_fft_pdata_tbl = "#PAL_FFT_PDATA_TBL" + gen_id
    data.save(pal_massive_fft_data_tbl)
    conn.create_table(table=pal_massive_fft_transformed_tbl, table_structure={'GROUP_ID': 'NVARCHAR(40)',
                                                                              'ID': 'INTEGER',
                                                                              'RE': 'DOUBLE',
                                                                              'IM': 'DOUBLE'})
    conn.create_table(table=pal_massive_fft_error_tbl, table_structure={'GROUP_ID': 'NVARCHAR(40)',
                                                                        'ERROR_TIMESTAMP': 'NVARCHAR(100)',
                                                                        'ERRORCODE': 'INTEGER',
                                                                        'ERROR_MESSAGE': 'NVARCHAR(200)'})
    create_dataframe_from_pandas(conn,
                                 pandas_df=pd.DataFrame({"POSITION": [1, 2, 3, 4],
                                                         "SCHEMA_NAME": [schema] * 4,
                                                         "TYPE_NAME": [pal_massive_fft_data_tbl,
                                                                       pal_massive_fft_param_tbl,
                                                                       pal_massive_fft_transformed_tbl,
                                                                       pal_massive_fft_error_tbl],
                                                         "PARAMETER_TYPE": ["IN", "IN", "OUT", "OUT"]}),
                                 table_name=pal_fft_pdata_tbl,
                                 disable_progressbar=True)
    try:
        conn.sql("CALL SYS.AFLLANG_WRAPPER_PROCEDURE_DROP('USER1', 'PAL_FFT_AFL_WRAPPER_MASS')")
    except dbapi.Error as db_err:
        pass
    try:
        conn.sql("CALL SYS.AFLLANG_WRAPPER_PROCEDURE_CREATE('AFLPAL', 'MASSIVE_FFT', {}, 'PAL_FFT_AFL_WRAPPER_MASS', {})".format(schema, pal_fft_pdata_tbl))
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, pal_massive_fft_data_tbl)
        try_drop(conn, pal_fft_pdata_tbl)
        raise
    param_rows = []
    if inverse is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'INVERSE', int(inverse), None, None))
    if num_type is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'NUMBER_TYPE', int(num_type), None, None))
    if window is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'WINDOW', None, None, window))
    if window_start is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'WINDOW_START', window_start, None, None))
    if window_length is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'WINDOW_LENGTH', window_length, None, None))
    if alpha is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'ALPHA', None, alpha, None))
    if beta is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'BETA', None, beta, None))
    if attenuation is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'ATTENUATION', None, attenuation, None))
    if r is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'R', None, attenuation, None))
    if flattop_model is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'MODE', None, None, flattop_model))
    if flattop_precision is not None:
        param_rows.append(('PAL_MASSIVE_PROCESSING_SPECIAL_GROUP_ID', 'PRECISION', None, None, flattop_precision))

    create_dataframe_from_pandas(conn,
                                 pandas_df=pd.DataFrame(param_rows, columns=['GROUP_ID',
                                                                             'NAME',
                                                                             'INTARGS',
                                                                             'DOUBLEARGS',
                                                                             'STRINGARGS']),
                                 table_name=pal_massive_fft_param_tbl,
                                 disable_progressbar=True)
    result_tbl = '#PAL_MASSIVE_FFT_RESULT_TBL_{}_{}'.format('1', gen_id)

    cursor = conn.connection.cursor()
    sql_block = """
    DO BEGIN
        t1 = SELECT * FROM {0};
        t2 = SELECT * FROM {1};
        CALL {2}.PAL_FFT_AFL_WRAPPER_MASS(:t1, :t2, t3, t4);
        CREATE LOCAL TEMPORARY TABLE {3}(GROUP_ID NVARCHAR(40),
                                         ID INTEGER,
                                         RE DOUBLE,
                                         IM DOUBLE);
        INSERT INTO {3} SELECT * FROM :t3;
    END
    """.format(pal_massive_fft_data_tbl,
               pal_massive_fft_param_tbl,
               schema,
               result_tbl)
    try:
        cursor.execute(sql_block)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, result_tbl)
        raise
    finally:
        try_drop(conn, pal_fft_pdata_tbl)
        try_drop(conn, pal_massive_fft_data_tbl)
        try_drop(conn, pal_massive_fft_param_tbl)
        try_drop(conn, pal_massive_fft_transformed_tbl)
        try_drop(conn, pal_massive_fft_error_tbl)
        cursor.close()
    return conn.table(result_tbl)
