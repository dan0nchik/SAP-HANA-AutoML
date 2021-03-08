"""
This module represents a visualizer for tree model and Graphviz is required.

Please download and install Graphviz from the following link:

https://graphviz.org/download/

The following class is available:

    * :class:`TreeModelDebriefing`
"""
import logging
import uuid
import json
from hdbcli import dbapi
try:
    import ipywidgets as widgets#pylint:disable=import-error
    from ipywidgets import interactive#pylint:disable=import-error
except ImportError as error:
    logging.getLogger(__name__).error("%s: %s", error.__class__.__name__, str(error))
    pass
try:
    from IPython.display import display, SVG, JSON#pylint:disable=import-error
except ImportError as error:
    logging.getLogger(__name__).error("%s: %s", error.__class__.__name__, str(error))
    pass
import pydotplus#pylint:disable=import-error
from graphviz import Source#pylint:disable=import-error
from hana_ml import dataframe
from hana_ml.ml_base import try_drop
from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    call_pal_auto
)
from hana_ml.visualizers.shap import ShapleyExplainer
logger = logging.getLogger(__name__)# pylint:disable=invalid-name


# pylint: disable=line-too-long
# pylint: disable=superfluous-parens
# pylint: disable=missing-class-docstring
class ModelDebriefingUtils(object):
    # Utility class for all model visualizations.
    def __init__(self):
        pass

    @staticmethod
    def is_pmml_format(data):
        """
        Determine whether the data format is PMML.

        Parameters
        ----------

        data : String

            Tree model data.

        Returns
        -------

        True or False

        """
        return data.endswith('</PMML>')

    @staticmethod
    def is_digraph_format(data):
        """
        Determine whether the data format is DOT.

        Parameters
        ----------

        data : String

            Tree model data.

        Returns
        -------

        True or False

        """
        return data.startswith('digraph')

    @staticmethod
    def visualize_with_xml(xml_data):
        """
        Print data in XML format.

        Parameters
        ----------

        xml_data : String

            Tree model data.

        Returns
        -------

        Print Data or Throw Exception

        """
        if ModelDebriefingUtils.is_pmml_format(xml_data):
            print(xml_data)
        else:
            raise TypeError('Data format is not PMML!')

    @staticmethod
    def visualize_with_json(json_data):
        """
        Visualize tree model by data in JSON format.

        Parameters
        ----------

        json_data : String

            Tree model data.

        Returns
        -------

        JSON Component or Throw Exception

            This object can be rendered by browser.
        """
        display(JSON(json.loads(json_data)))

    @staticmethod
    def visualize_with_dot(dot_data):
        """
        Visualize tree model by data in DOT format.

        Parameters
        ----------

        dot_data : String

            Tree model data.

        Returns
        -------

        Visualize SVG Component or Throw Exception

            This object can be rendered by browser.
        """
        graph = pydotplus.graph_from_dot_data(dot_data.encode('utf8'))
        graph_size = len(graph.get_nodes())
        if graph_size <= 1001:
            ModelDebriefingUtils.warning('You can also export the model as a DOT file and use the ZGRViewer tool to view the model.')
            display(SVG(Source(dot_data).pipe(format='svg')))
        else:
            ModelDebriefingUtils.warning('There are too many tree nodes[{}].\n'
                                         'You can export the model as a DOT file and use the ZGRViewer tool to view the model.'.format(str(graph_size)))

    @staticmethod
    def visualize(data):
        """
        Visualize tree model by data in DOT, JSON or XML format.

        Parameters
        ----------

        data : String

            Tree model data.

        Returns
        -------

        SVG, JSON, XML Component or Throw Exception

            This object can be rendered by browser.
        """
        if ModelDebriefingUtils.is_digraph_format(data):
            ModelDebriefingUtils.visualize_with_dot(data)
        elif ModelDebriefingUtils.is_pmml_format(data):
            ModelDebriefingUtils.visualize_with_xml(data)
        else:
            ModelDebriefingUtils.visualize_with_json(data)

    @staticmethod
    def visualize_from_file(path):
        """
        Visualize tree model by a DOT, JSON or XML file.

        Parameters
        ----------

        path : String

            File path.

        Returns
        -------

        SVG, JSON or XML Component

            This object can be rendered by browser.
        """
        return ModelDebriefingUtils.visualize(ModelDebriefingUtils.read_file(path))

    @staticmethod
    def save_file(str_content, path):
        """
        Save tree model as a file.

        Parameters
        ----------

        str_content : String

            Tree model content.

        path : String

            File path.

        """
        temp_file = open(path, "w")
        temp_file.write(str_content)
        temp_file.close()

    @staticmethod
    def read_file(path):
        """
        Read tree model from a file.

        Parameters
        ----------

        path : String

            File path.

        Returns
        -------

        String

            The tree model content.
        """
        fd = open(path, 'r')#pylint: disable=invalid-name
        data = fd.read()
        fd.close()
        return data

    @staticmethod
    def check_dataframe_type(model):
        """
        Determine whether the type of model object is hana_ml.dataframe.DataFrame.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        Throw exception

        """
        if isinstance(model, dataframe.DataFrame) is False:
            raise TypeError("The type of parameter 'model' must be hana_ml.dataframe.DataFrame!")

    @staticmethod
    def warning(msg):
        """
        Print message with red color.

        Parameters
        ----------

        msg : String

            Message.
        """
        print('\033[31m{}'.format(msg))


# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=superfluous-parens
class TreeModelDebriefing(object):
    """
    Visualize tree model.
        - Dependency packages
            1.Graphviz
                To render the generated DOT source code, you need to install Graphviz.\n
                Download page: https://www.graphviz.org/download/.\n
                Make sure that the directory containing the dot executable is on your system path.
            2.graphviz \n
            3.pydotplus \n
            4.ipywidgets
                To render the Jupyter widgets, you also need to install JupyterLab extension.\n
                Install Page: https://ipywidgets.readthedocs.io/en/latest/user_install.

    Examples
    --------

    Create TreeModelDebriefing Instance and the model is stored in the table PAL_DT_MODEL_TBL:

    >>> treeModelDebriefing = TreeModelDebriefing()

    Visualize Tree Model in JSON format:

    >>> treeModelDebriefing.tree_debrief(PAL_DT_MODEL_TBL)

    .. image:: json_model.png

    Visualize Tree Model in DOT format:

    >>> treeModelDebriefing.tree_parse(PAL_DT_MODEL_TBL)
    >>> treeModelDebriefing.tree_debrief_with_dot(PAL_DT_MODEL_TBL)

    .. image:: dot_model.png

    Visualize Tree Model in XML format the model is stored in the table PAL_RDT_MODEL_TBL:

    >>> treeModelDebriefing.tree_debrief(PAL_RDT_MODEL_TBL)

    .. image:: xml_model.png


    """

    def __init__(self):
        pass

    __PARSE_MODEL_FUC_NAME = 'PAL_VISUALIZE_MODEL'

    __TREE_INDEX_FIRST_VALUE = '0'
    __TREE_INDEX_RANGE_MSG = 'The lowest input value is {}, and the highest is {}.\nPlease check your input value!'
    __TREE_DICT_NAME = '_tree_dict'
    __TREE_DOT_DICT_NAME = '_tree_dot_dict'

    __VISUALIZE_FUNC_NAME = '_visualize_func'
    __VISUALIZE_DOT_FUNC_NAME = '_visualize_dot_func'

    __EXPORT_FUNC_NAME = '_export_func'
    __EXPORT_DOT_FUNC_NAME = '_export_dot_func'
    __EXPORT_PARAMETER = {'manual': True, 'manual_name': 'Download Model File'}

    __API_MSG = 'You must parse the model firstly!'
    __ENTER_FILE_NAME_MSG = 'Please enter the correct file name!'

    def __add_tree_dict(self, model):
        if model.__dict__.get(self.__TREE_DICT_NAME) is None:
            model.__dict__[self.__TREE_DICT_NAME] = self.__parse_tree_model(model)

    def __check_tree_dot_dict(self, model):
        if model.__dict__.get(self.__TREE_DOT_DICT_NAME) is None:
            logger.error(self.__API_MSG)
            raise AttributeError(self.__API_MSG)

    @staticmethod
    def __create_tree_index_widget(min_tree_index_value, max_tree_index_value):
        return widgets.Text(value=str(min_tree_index_value), placeholder='{} to {}'.format(min_tree_index_value, max_tree_index_value), description='Tree Index:', disabled=False)

    @staticmethod
    def __create_file_name_widget():
        return widgets.Text(value='', placeholder='Enter Name...', description='Model File:', disabled=False)

    def __parse_tree_model(self, model):
        tree_dict = {}
        if len(model.columns) == 3:
            # multiple trees
            # |ROW_INDEX|TREE_INDEX|MODEL_CONTENT|
            for tree_index, single_tree_list in \
                    model.sort(model.columns[0]).collect().groupby(model.columns[1])[model.columns[2]]:
                tree_dict[str(tree_index)] = "".join(single_tree_list)
        else:
            # single tree
            # |ROW_INDEX|MODEL_CONTENT|
            tree_dict[self.__TREE_INDEX_FIRST_VALUE] = "".join(model.sort(model.columns[0]).collect()[model.columns[1]])

        return tree_dict

    def __add_interact_visualize_func(self, model):
        if model.__dict__.get(self.__VISUALIZE_FUNC_NAME) is None:
            tree_list = list(model.__dict__[self.__TREE_DICT_NAME].keys())
            tree_size = len(tree_list)

            min_value = int(tree_list[0])
            max_value = min_value
            if tree_size > 1:
                max_value = int(tree_list[tree_size - 1])

            def visualize(tree_index):
                try:
                    tree_index = int(tree_index)
                    if min_value <= tree_index <= max_value:
                        data = model.__dict__[self.__TREE_DICT_NAME][str(tree_index)]
                        if ModelDebriefingUtils.is_pmml_format(data):
                            ModelDebriefingUtils.visualize_with_xml(data)
                        else:
                            ModelDebriefingUtils.visualize_with_json(data)
                    else:
                        ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))
                except ValueError:
                    ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))

            model.__dict__[self.__VISUALIZE_FUNC_NAME] = interactive(visualize,
                                                                     tree_index=self.__create_tree_index_widget(min_value, max_value))

    def __add_interact_visualize_dot_func(self, model):
        if model.__dict__.get(self.__VISUALIZE_DOT_FUNC_NAME) is None:
            tree_list = list(model.__dict__[self.__TREE_DOT_DICT_NAME].keys())
            tree_size = len(tree_list)

            min_value = int(tree_list[0])
            max_value = min_value
            if tree_size > 1:
                max_value = int(tree_list[tree_size - 1])

            def visualize(tree_index):
                try:
                    tree_index = int(tree_index)
                    if min_value <= tree_index <= max_value:
                        ModelDebriefingUtils.visualize_with_dot(model.__dict__[self.__TREE_DOT_DICT_NAME][str(tree_index)])
                    else:
                        ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))
                except ValueError:
                    ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))

            model.__dict__[self.__VISUALIZE_DOT_FUNC_NAME] = interactive(visualize,
                                                                         tree_index=self.__create_tree_index_widget(min_value, max_value))

    def tree_debrief(self, model):
        """
        Visualize tree model by data in JSON or XML format.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        JSON or XML Component

            This object can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        self.__add_tree_dict(model)

        if len(model.columns) == 3:
            # multiple trees
            self.__add_interact_visualize_func(model)
            display(model.__dict__[self.__VISUALIZE_FUNC_NAME])
        else:
            # single tree
            data = model.__dict__[self.__TREE_DICT_NAME][self.__TREE_INDEX_FIRST_VALUE]
            if ModelDebriefingUtils.is_pmml_format(data):
                ModelDebriefingUtils.visualize_with_xml(data)
            else:
                ModelDebriefingUtils.visualize_with_json(data)

    def tree_debrief_with_dot(self, model):
        """
        Visualize tree model by data in DOT format.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        SVG Component

            This object can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        self.__check_tree_dot_dict(model)

        if len(model.columns) == 3:
            # multiple trees
            self.__add_interact_visualize_dot_func(model)
            display(model.__dict__[self.__VISUALIZE_DOT_FUNC_NAME])
        else:
            # single tree
            ModelDebriefingUtils.visualize_with_dot(model.__dict__[self.__TREE_DOT_DICT_NAME][self.__TREE_INDEX_FIRST_VALUE])

    def tree_parse(self, model):
        """
        Transform tree model content using DOT language.

        Parameters
        ----------

        model : DataFrame

            Tree model.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        dot_tbl_name = '#PAL_DT_DOT_TBL_{}'.format(str(uuid.uuid1()).replace('-', '_').upper())
        tables = [dot_tbl_name]
        param_rows = []
        try:
            call_pal_auto(model.connection_context,
                          self.__PARSE_MODEL_FUC_NAME,
                          model,
                          ParameterTable().with_data(param_rows),
                          *tables)

            model.__dict__[self.__TREE_DOT_DICT_NAME] = self.__parse_tree_model(model.connection_context.table(dot_tbl_name))
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(model.connection_context, dot_tbl_name)
            raise

    def __add_interact_export_func(self, model):
        if model.__dict__.get(self.__EXPORT_FUNC_NAME) is None:
            if len(model.columns) == 3:
                # multiple trees
                tree_list = list(model.__dict__[self.__TREE_DICT_NAME].keys())
                tree_size = len(tree_list)

                min_value = int(tree_list[0])
                max_value = min_value
                if tree_size > 1:
                    max_value = int(tree_list[tree_size - 1])

                def download(file_name, tree_index):
                    print('')
                    try:
                        file_name = str(file_name).strip()
                        if file_name == '':
                            ModelDebriefingUtils.warning(self.__ENTER_FILE_NAME_MSG)
                        else:
                            tree_index = int(tree_index)
                            if min_value <= tree_index <= max_value:
                                data = model.__dict__[self.__TREE_DICT_NAME][str(tree_index)]
                                if ModelDebriefingUtils.is_pmml_format(data):
                                    ModelDebriefingUtils.save_file(data, '{}_{}.xml'.format(file_name, str(tree_index)))
                                else:
                                    ModelDebriefingUtils.save_file(data, '{}_{}.json'.format(file_name, str(tree_index)))
                            else:
                                ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))
                    except ValueError:
                        ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))

                model.__dict__[self.__EXPORT_FUNC_NAME] = interactive(download,
                                                                      self.__EXPORT_PARAMETER,
                                                                      file_name=self.__create_file_name_widget(), tree_index=self.__create_tree_index_widget(min_value, max_value))
            else:
                # single tree
                def download(file_name):
                    print('')
                    file_name = str(file_name).strip()
                    if file_name == '':
                        ModelDebriefingUtils.warning(self.__ENTER_FILE_NAME_MSG)
                    else:
                        data = model.__dict__[self.__TREE_DICT_NAME][self.__TREE_INDEX_FIRST_VALUE]
                        if ModelDebriefingUtils.is_pmml_format(data):
                            ModelDebriefingUtils.save_file(data, '{}.xml'.format(file_name))
                        else:
                            ModelDebriefingUtils.save_file(data, '{}.json'.format(file_name))

                model.__dict__[self.__EXPORT_FUNC_NAME] = interactive(download,
                                                                      self.__EXPORT_PARAMETER,
                                                                      file_name=self.__create_file_name_widget())

    def __add_interact_export_dot_func(self, model):
        if model.__dict__.get(self.__EXPORT_DOT_FUNC_NAME) is None:
            if len(model.columns) == 3:
                # multiple trees
                tree_list = list(model.__dict__[self.__TREE_DOT_DICT_NAME].keys())
                tree_size = len(tree_list)

                min_value = int(tree_list[0])
                max_value = min_value
                if tree_size > 1:
                    max_value = int(tree_list[tree_size - 1])

                def download(file_name, tree_index):
                    print('')
                    try:
                        file_name = str(file_name).strip()
                        if file_name == '':
                            ModelDebriefingUtils.warning(self.__ENTER_FILE_NAME_MSG)
                        else:
                            tree_index = int(tree_index)
                            if min_value <= tree_index <= max_value:
                                ModelDebriefingUtils.save_file(model.__dict__[self.__TREE_DOT_DICT_NAME][str(tree_index)],
                                                               '{}_{}.dot'.format(file_name, tree_index))
                            else:
                                ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))
                    except ValueError:
                        ModelDebriefingUtils.warning(self.__TREE_INDEX_RANGE_MSG.format(min_value, max_value))

                model.__dict__[self.__EXPORT_DOT_FUNC_NAME] = interactive(download,
                                                                          self.__EXPORT_PARAMETER,
                                                                          file_name=self.__create_file_name_widget(), tree_index=self.__create_tree_index_widget(min_value, max_value))
            else:
                # single tree
                def download(file_name):
                    print('')
                    file_name = str(file_name).strip()
                    if file_name == '':
                        ModelDebriefingUtils.warning(self.__ENTER_FILE_NAME_MSG)
                    else:
                        ModelDebriefingUtils.save_file(model.__dict__[self.__TREE_DOT_DICT_NAME][self.__TREE_INDEX_FIRST_VALUE], file_name + '.dot')

                model.__dict__[self.__EXPORT_DOT_FUNC_NAME] = interactive(download,
                                                                          self.__EXPORT_PARAMETER,
                                                                          file_name=self.__create_file_name_widget())

    def tree_export(self, model):
        """
        Export tree model as a JSON or XML file.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        Interactive Text and Button Widgets

            Those widgets can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        self.__add_tree_dict(model)
        self.__add_interact_export_func(model)
        display(model.__dict__[self.__EXPORT_FUNC_NAME])

    def tree_export_with_dot(self, model):
        """
        Export tree model as a DOT file.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        Interactive Text and Button Widgets

            Those widgets can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        self.__check_tree_dot_dict(model)
        self.__add_interact_export_dot_func(model)
        display(model.__dict__[self.__EXPORT_DOT_FUNC_NAME])

    @staticmethod
    def tree_debrief_from_file(path):
        """
        Visualize tree model by a DOT, JSON or XML file.

        Parameters
        ----------

        path : String

            File path.

        Returns
        -------

        SVG, JSON or XML Component

            This object can be rendered by browser.
        """
        return ModelDebriefingUtils.visualize_from_file(path)

    @staticmethod
    def shapley_explainer(predict_result: dataframe.DataFrame, predict_data: dataframe.DataFrame, key, label):
        """
        Create Shapley explainer to explain the output of machine learning model. \n
        It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

        Parameters
        ----------
        predict_result : DataFrame
            Predicted result, structured as follows:
                -  1st column : ID
                -  2nd column : SCORE, i.e. predicted class label
                -  3rd column : CONFIDENCE, i.e. confidence value for the assigned class label
                -  4th column : REASON CODE, valid only for tree-based functionalities.

        predict_data : DataFrame
            Predicted dataset.

        key : str
            Name of the ID column.

        label : str
            Name of the dependent variable.

        Returns
        -------
        :class:`~hana_ml.visualizers.shap.ShapleyExplainer`
            Shapley explainer.
        """
        return ShapleyExplainer(predict_result, predict_data, key, label)
