"""
This module contains Python wrapper for PAL conditional random field(CRF) algorithm.

The following class is available:

    * :class:`CRF`
"""

# pylint:disable=line-too-long, too-many-instance-attributes, too-few-public-methods
import logging
import uuid

from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
#from hana_ml.dataframe import quotename
from .pal_base import (
    PALBase,
    #Table,
    ParameterTable,
    #NVARCHAR,
    #parse_one_dtype,
    pal_param_register,
    try_drop,
    require_pal_usable,
    call_pal_auto
)

logger = logging.getLogger(__name__)#pylint:disable=invalid-name

class CRF(PALBase):
    """
    Conditional random field(CRF) for labeling and segmenting sequence data(e.g. text).

    Parameters
    ----------

    epsilon : float, optional

        Convergence tolerance of the optimization algorithm.

        Defaults to 1e-4.

    lamb : float, optional

        Regularization weight, should be greater than 0.

        Defaults t0 1.0.

    max_iter : int, optional

        Maximum number of iterations in optimization.

        Defaults to 1000.

    lbfgs_m : int, optional

        Number of memories to be stored in L_BFGS optimization algorithm.

        Defaults to 25.

    use_class_feature : bool, optional

        To include a feature for class/label.
        This is the same as having a bias vector in a model.

        Defaults to True.

    use_word : bool, optional

        If True, gives you feature for current word.

        Defaults to True.

    use_ngrams : bool, optional

        Whether to make feature from letter n-grams, i.e. substrings of the word.

        Defaults to True.

    mid_ngrams : bool, optional

        Whether to include character n-gram features for n-grams that
        contain neither the beginning or the end of the word.

        Defaults to False.

    max_ngram_length : int, optional

        Upper limit for the size of n-grams to be included.
        Effective only this parameter is positive.

    use_prev : bool, optional

        Whether or not to include a feature for previous word and current word,
        and together with other options enables other previous features.

        Defaults to True.

    use_next : bool, optional

        Whether or not to include a feature for next word and current word.

        Defaults to True.

    disjunction_width : int, optional

        Defines the width for disjunctions of words, see ``use_disjunctive``.

        Defaults to 4.

    use_disjunctive : bool, optional

        Whether or not to include in features giving disjunctions of words
        anywhere in left or right ``disjunction_width`` words.

        Defaults to True.

    use_seqs : bool, optional

        Whether or not to use any class combination features.

        Defaults to True.

    use_prev_seqs : bool, optional

        Whether or not to use any class combination features using the previous class.

        Defaults to True.

    use_type_seqs : bool, optional

        Whther or not to use basic zeroth order word shape features.

        Defaults to True.

    use_type_seqs2 : bool, optional

        Whether or not to add additional first and second order word shape features.

        Defaults to True.

    use_type_yseqs : bool, optional

        Whether or not to use some first order word shape patterns.

        Defaults to True.

    word_shape : int, optional

        Word shape, e.g. whether capitalized or numeric.
        Only supports chris2UseLC currently.
        Do not use word shape if this is 0.

    thread_ratio : float, optional

        Specifies the ratio of total number of threads that can be used by
        the fit(i.e. training) function.

        The range of this parameter is from 0 to 1.

        0 means only using single thread, 1 means using at most all
        available threads currently.

        Values outside this range are ignored, and the fit function
        heuristically determines the number of threads to use.

        Defaults to 1.0.

    Attributes
    ----------

    model_ : DataFrame

        CRF model content.

    stats_ : DataFrame

        Statistic info for CRF model fitting, structured as follows:

            - 1st column: name of the statistics, type NVARCHAR(100).
            - 2nd column: the corresponding statistics value, type NVARCHAR(1000).

    optimal_param_ : DataFrame

        Placeholder for storing optimal parameter of the model.
        None empty only when parameter selection is triggered (in the future).

    Examples
    --------

    Input data for training:

    >>> df.head(10).collect()
       DOC_ID  WORD_POSITION      WORD LABEL
    0       1              1    RECORD     O
    1       1              2   #497321     O
    2       1              3  78554939     O
    3       1              4         |     O
    4       1              5       LRH     O
    5       1              6         |     O
    6       1              7  62413233     O
    7       1              8         |     O
    8       1              9         |     O
    9       1             10   7368393     O

    Set up an instance of CRF model, and fit it on the training data:

    >>> crf = CRF(lamb=0.1,
    ...           max_iter=1000,
    ...           epsilon=1e-4,
    ...           lbfgs_m=25,
    ...           word_shape=0,
    ...           thread_ratio=1.0)
    >>> crf.fit(data=df, doc_id="DOC_ID", word_pos="WORD_POSITION",
    ...         word="WORD", label="LABEL")

    Check the trained CRF model and related statistics:

    >>> crf.model_.collect()
       ROW_INDEX                                      MODEL_CONTENT
    0          0  {"classIndex":[["O","OxygenSaturation"]],"defa...
    >>> crf.stats_.head(10).collect()
             STAT_NAME           STAT_VALUE
    0              obj  0.44251900977373015
    1             iter                   22
    2  solution status            Converged
    3      numSentence                    2
    4          numWord                   92
    5      numFeatures                  963
    6           iter 1          obj=26.6557
    7           iter 2          obj=14.8484
    8           iter 3          obj=5.36967
    9           iter 4           obj=2.4382

    Input data for predicting labels using trained CRF model

    >>> df_pred.head(10).collect()
       DOC_ID  WORD_POSITION         WORD
    0       2              1      GENERAL
    1       2              2     PHYSICAL
    2       2              3  EXAMINATION
    3       2              4            :
    4       2              5        VITAL
    5       2              6        SIGNS
    6       2              7            :
    7       2              8        Blood
    8       2              9     pressure
    9       2             10        86g52

    Do the prediction:

    >>> res = crf.predict(data=df_pred, doc_id='DOC_ID', word_pos='WORD_POSITION',
    ...                   word='WORD', thread_ratio=1.0)

    Check the prediction result:

    >>> df_pred.head(10).collect()
       DOC_ID  WORD_POSITION         WORD
    0       2              1      GENERAL
    1       2              2     PHYSICAL
    2       2              3  EXAMINATION
    3       2              4            :
    4       2              5        VITAL
    5       2              6        SIGNS
    6       2              7            :
    7       2              8        Blood
    8       2              9     pressure
    9       2             10        86g52
    """
    #pylint:disable=too-many-arguments, too-many-locals
    def __init__(self,
                 lamb=None,
                 epsilon=None,
                 max_iter=None,
                 lbfgs_m=None,
                 use_class_feature=None,
                 use_word=None,
                 use_ngrams=None,
                 mid_ngrams=False,
                 max_ngram_length=None,
                 use_prev=None,
                 use_next=None,
                 disjunction_width=None,
                 use_disjunctive=None,
                 use_seqs=None,
                 use_prev_seqs=None,
                 use_type_seqs=None,
                 use_type_seqs2=None,
                 use_type_yseqs=None,
                 word_shape=None,
                 thread_ratio=None):
        super(CRF, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.lamb = self._arg('lamb', lamb, float)
        self.epsilon = self._arg('epsilon', epsilon, float)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.lbfgs_m = self._arg('lbfgs_m', lbfgs_m, int)
        self.use_class_feature = self._arg('use_class_feature',
                                           use_class_feature, bool)
        self.use_word = self._arg('use_word', use_word, bool)
        self.use_ngrams = self._arg('use_ngrams', use_ngrams, bool)
        self.mid_ngrams = self._arg('mid_ngrams', mid_ngrams, bool)
        self.max_ngram_length = self._arg('max_ngram_length', max_ngram_length, int)
        self.use_next = self._arg('use_next', use_next, bool)
        self.use_prev = self._arg('use_prev', use_prev, bool)
        self.disjunction_width = self._arg('disjunction_width',
                                           disjunction_width, bool)
        self.use_disjunctive = self._arg('use_disjunctive',
                                         use_disjunctive, bool)
        self.use_seqs = self._arg('use_seqs', use_seqs, bool)
        self.use_prev_seqs = self._arg('use_prev_seqs', use_prev_seqs, bool)
        self.use_type_seqs = self._arg('use_type_seqs', use_type_seqs, bool)
        self.use_type_seqs2 = self._arg('use_type_seqs2', use_type_seqs2, bool)
        self.use_type_yseqs = self._arg('use_type_yseqs', use_type_yseqs, bool)
        #word-shape parameter may needs more consideration.
        self.word_shape = self._arg('word_shape', word_shape, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)


    def fit(self, data, doc_id=None, word_pos=None,
            word=None, label=None):
        """
        Function for training the CRF model on English text.

        Parameters
        ----------

        data : DataFrame

            Input data for training/fitting the CRF model.

            It should contain at least 4 columns, corresponding to document ID,
            word position, word and label, respectively.

        doc_id : str, optional

            Name of the column for document ID.

            Defaults to the first column of the input data.

        word_pos : str, optional

            Name of the column for word position.

            Defaults to the second column of the input data.

        word : str, optional

            Name of the column for word.

            Defaults to the third column of the input data.

        label : str, optional

            Name of the label column.

            Defaults to the final column of the input data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        if len(cols) < 4:
            msg = ("Input data contains only {} columns, ".format(len(cols))+
                   "while CRF model fitting requires at least 4.")
            logger.error(msg)
            raise ValueError(msg)
        doc_id = self._arg('doc_id', doc_id, str)
        word_pos = self._arg('word_pos', word_pos, str)
        word = self._arg('word', word, str)
        label = self._arg('label', label, str)
        if doc_id is None:
            doc_id = cols[0]
        if word_pos is None:
            word_pos = cols[1]
        if word is None:
            word = cols[2]
        if label is None:
            label = cols[-1]
        used_cols = [doc_id, word_pos, word, label]
        data_ = data[used_cols]
        param_rows = [('ENET_LAMBDA', None, self.lamb, None),
                      ('EXIT_THRESHOLD', None, self.epsilon, None),
                      ('MAX_ITERATION', self.max_iter, None, None),
                      ('LBFGS_M', self.lbfgs_m, None, None),
                      ('USE_CLASS_FEATURE', self.use_class_feature, None, None),
                      ('USE_WORD', self.use_word, None, None),
                      ('USE_NGRAMS', self.use_ngrams, None, None),
                      ('NO-MIDNGRAMS', not self.mid_ngrams, None, None),
                      ('MAX_NGRAM_LENGTH', self.max_ngram_length, None, None),
                      ('USE_PREV', self.use_prev, None, None),
                      ('USE_NEXT', self.use_next, None, None),
                      ('USE_DISJUNCTIVE', self.use_disjunctive, None, None),
                      ('DISJUNCTION_WIDTH', self.disjunction_width, None, None),
                      ('USE_SEQUENCES', self.use_seqs, None, None),
                      ('USE_PREVSEQUENCES', self.use_prev_seqs, None, None),
                      ('USE_TYPE_SEQS', self.use_type_seqs, None, None),
                      ('USE_TYPE_SEQS2', self.use_type_seqs2, None, None),
                      ('USE_TYPE_YSEQUENCES', self.use_type_yseqs, None, None),
                      ('WORD_SHAPE', self.word_shape, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_')
        tables = ['MODEL', 'STATS', 'OPTIMAL_PARAM']
        tables = ["#PAL_CRF_{}_TBL_{}_{}".format(table, self.id, unique_id)
                  for table in tables]
        model_tbl, stats_tbl, optim_param_tbl = tables
        try:
            call_pal_auto(conn,
                          'PAL_CRF',
                          data_,
                          ParameterTable().with_data(param_rows),
                          model_tbl,
                          stats_tbl,
                          optim_param_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        #pylint:disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.stats_ = conn.table(stats_tbl)
        self.optim_param_ = conn.table(optim_param_tbl)

    #pylint:disable=too-many-arguments
    def predict(self,
                data,
                doc_id=None,
                word_pos=None,
                word=None,
                thread_ratio=None):
        """
        The function that predicts text labels based trained CRF model.

        Parameters
        ------------
        data : DataFrame
            Input data to predict the labels.
            It should contain at least 3 columns, corresponding to document ID,
            word position and word, respectively.

        doc_id : str, optional
            Name of the column for document ID.

            Defaults to the first column of the input data.

        word_pos : str, optional
            Name of the column for word position.

            Defaults to the second column of the input data.

        word : str, optional
            Name of the column for word.

            Defaults to the third column of the input data.

        thread_ratio : float, optional
            Specifies the ratio of total number of threads that can be used
            by predict function.

            The range of this parameter is from 0 to 1.

            0 means only using a single thread, and 1 means using at most all
            available threads currently.

            Values outside this range are ignored, and predict function
            heuristically determines the number of threads to use.

            Defaults to 1.0.

        Returns
        -------
        DataFrame
            Prediction result for the input data, structured as follows:

                - 1st column: document ID,
                - 2nd column: word position,
                - 3rd column: label.

        """
        conn = data.connection_context
        if not hasattr(self, 'model_'):
            msg = ("Model not initialized. Perform a fit first.")
            raise FitIncompleteError(msg)
        cols = data.columns
        if len(cols) < 3:
            msg = ("Input data contains only {} columns, ".format(len(cols))+
                   "while CRF label prediction requires at least 3.")
            logger.error(msg)
            raise ValueError(msg)
        doc_id = self._arg('doc_id', doc_id, str)
        word_pos = self._arg('word_pos', word_pos, str)
        word = self._arg('word', word, str)
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        if doc_id is None:
            doc_id = cols[0]
        if word_pos is None:
            word_pos = cols[1]
        if word is None:
            word = cols[2]
        used_cols = [doc_id, word_pos, word]
        data_ = data[used_cols]
        param_rows = [('THREAD_RATIO', None, thread_ratio, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_CRF_INFERENCE_RESULT_TBL_{}_{}".format(self.id, unique_id)
        #tables = ['DATA', 'MODEL', 'PARAM', 'RESULT']
        #tables = ["#PAL_CRF_INFERENCE_{}_TBL_{}_{}".format(tbl, self.id, unique_id)
        #          for tbl in tables]
        #data_tbl, model_tbl, param_tbl, result_tbl = tables
        #result_spec = [
        #    (parse_one_dtype(data.dtypes([doc_id])[0])),
        #    (parse_one_dtype(data.dtypes([word_pos])[0])),
        #    ('LABEL', NVARCHAR(500))
        #]
        #model_tbl = "#PAL_CRF_MODEL_TBL_{}_{}".format(self.id, unique_id)
        #with conn.connection.cursor() as cur:
        #    cur.execute('CREATE LOCAL TEMPORARY COLUMN TABLE {}("ID" INTEGER, "CONTENT" NCLOB)'.format(quotename(model_tbl)))#pylint:disable=line-too-long
        #    cur.execute('INSERT INTO {}(ID, CONTENT) SELECT ROW_INDEX, MODEL_CONTENT FROM {} WHERE 1=1'.format(quotename(model_tbl), self.model_._ttab_reference))#pylint:disable=line-too-long, protected-access
        try:
            #self._materialize(data_tbl, data)
            #self._materialize(model_tbl, model_)
            #self._create(ParameterTable(param_tbl).with_data(param_rows))
            #self._create(Table(result_tbl, result_spec))
            #self._call_pal('PAL_CRF_INFERENCE', *tables)
            call_pal_auto(conn,
                          'PAL_CRF_INFERENCE',
                          data_,
                          self.model_,
                          ParameterTable().with_data(param_rows),
                          result_tbl)

        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            #self._try_drop(tables)
            #self._try_drop(model_tbl)
            try_drop(conn, result_tbl)

            raise
        #self._try_drop(model_tbl)
        return conn.table(result_tbl)
