"""
This module handles generation of all AMDP related artifacts based on the provided
consumption layer elements. Currently this is experimental code only.

The following class is available:

    * :class:`AMDPGenerator`

"""
#pylint: disable=line-too-long, too-many-locals, too-many-nested-blocks
#pylint: disable=too-many-branches, too-many-statements, consider-using-in, invalid-name
#pylint: disable=inconsistent-return-statements, too-few-public-methods, unused-variable
#pylint: disable=useless-object-inheritance
import os
import re
from hana_ml.artifacts.generators.filewriter.abap import AMDPWriter

from hana_ml.artifacts.config import ConfigConstants, ConfigHandler
from hana_ml.artifacts.utils import DirectoryHandler

from hana_ml.artifacts.generators.sql_processor import SqlProcessor



class AMDPGenerator(object):
    """
    This class provides AMDP specific generation functionality. It also extend the config
    to cater for AMDP generation specific config.

    Parameters
    ----------
    project_name : str
        Name of the project.

    version : str
        Version.

    connection_context : str
        The connection to the SAP HANA.

    outputdir : str
        The directory of output.

    """

    def __init__(self, project_name, version, connection_context, outputdir):
        self.directory_handler = DirectoryHandler()
        self.config = ConfigHandler.init_config(project_name, version, None, outputdir)
        sql_processor = SqlProcessor(self.config)
        sql_processor.parse_sql_trace(connection_context)
        self._extend_config()

    def _build_folder_structure(self):
        """
        Build up the folder structure. It is currenlty not a deep structure but just a subbfolder abap
        under the root output path.
        """
        # self._clean_folder_structure()
        # Create base directories
        self.directory_handler.create_directory(self.config.get_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH_ABAP))

    def _clean_folder_structure(self):
        """
        Clean up physical folder structure.
        """
        path = self.config.get_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH_ABAP)
        if os.path.exists(path):
            self.directory_handler.delete_directory_content(path)
            os.rmdir(path)

    def _extend_config(self):
        """
        Extend the config to cater for AMDP generation specific config.
        """
        output_path_amdp = os.path.join(self.config.get_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH),
                                        ConfigConstants.ABAP_BASE_PATH)
        self.config.add_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH_ABAP, output_path_amdp)

    def generate(self, training_dataset='', apply_dataset='', no_reason_features=3):
        """
        Generate the artifacts by first building up the required folder structure for artifacts storage and then
        generating the different required files.

        Parameters
        ----------
        training_dataset : str, optional
            Name of training dataset.

            Defaults to ''.
        apply_dataset : str, optional
            Name of apply dataset.

            Defaults to ''.
        no_reason_features : int, optional
            Number of features in reason code to display.

            Defaults to 3.

        """
        self._build_folder_structure()

        amdp_writer = AMDPWriter(self.config)
        sql_key_sql = SqlProcessor.TRACE_KEY_SQL_PROCESSED
        error_message = ''
        sql_processed_cons_layer = self.config.get_entry(ConfigConstants.CONFIG_KEY_SQL_PROCESSED)[
            SqlProcessor.TRACE_KEY_CONSUMPTION_LAYER]
        # base layer also needed to construct the abap class as it contains both layers at once
        sql_processed_base_layer = self.config.get_entry(ConfigConstants.CONFIG_KEY_SQL_PROCESSED)[
            SqlProcessor.TRACE_KEY_BASE_LAYER]

        calls_grouped_by_algorithm = {}
        for element in sql_processed_cons_layer:
            if calls_grouped_by_algorithm.get(element['algo']) is None:
                calls_grouped_by_algorithm[element['algo']] = [element]
            else:
                calls_grouped_by_algorithm[element['algo']].append(element)
        for algo, classifier in calls_grouped_by_algorithm.items():
            # Support check
            if not any([supp_algo in algo.lower() for supp_algo in ConfigConstants.AMDP_SUPPORTED_ALGORITHMS]):
                error_message += 'Algorithm \'{}\' not yet supported for automated hemi generation'.format(algo)
                continue
            bodys, fit_input, predict_input, fit_output, predict_output = [], None, None, None, None
            for element in classifier:
                if not isinstance(element, dict):
                    continue  # ignore
                if element['groups'][0]['type'] in {'fit', 'predict'}:
                    if sql_key_sql in element:
                        if 'output' in element[sql_key_sql]:
                            for table in element[sql_key_sql]['output']:
                                if self.config.is_model_category(table['cat']) or self.config.is_fitted_category(
                                        table['cat']):
                                    if element['groups'][0]['type'] == 'predict':
                                        predict_output = table

                        if 'input' in element[sql_key_sql]:
                            for table in element[sql_key_sql]['input']:
                                if not self.config.is_model_category(table['cat']):
                                    if element['groups'][0]['type'] == 'fit':
                                        fit_input = table  # Only one output allowed in transformation context
                                    else:
                                        predict_input = table

                        if 'body' in element[sql_key_sql]:
                            item = element[sql_key_sql]['body'][0]  # Intermediate step for readability of next line
                            bodys.append(item[sql_key_sql].format(*item['sql_vars']))
                layer = sql_processed_base_layer[algo]
                if not isinstance(layer, dict):
                    error_message += 'No corresponding base layer found for algorithm:' + str(algo) + '\n'
                    break
                if 'fit' in layer.keys() and 'predict' in layer.keys() and \
                        'sql' in layer['fit'].keys() and 'sql' in layer['predict']:
                    fit_base = layer['fit']
                    predict_base = layer['predict']
                    fit_param_sql = self._extract_params_definition_from_sql(fit_base['sql'])
                    predict_param_sql = self._extract_params_definition_from_sql(predict_base['sql'])

                    pal_fit_sql = self._find_pal_function_from_sql(fit_base['sql'])
                    for synonym in fit_base['synonyms']:
                        pal_fit_sql = pal_fit_sql.replace(synonym['synonym'],
                                                          (synonym['schema'] + '.' if synonym['schema'] else '') +
                                                          synonym['object'])
                    pal_predict_sql = self._find_pal_function_from_sql(predict_base['sql'])
                    for synonym in predict_base['synonyms']:
                        pal_predict_sql = pal_predict_sql.replace(synonym['synonym'],
                                                                  (synonym['schema'] + '.' if synonym['schema'] else '') +
                                                                  synonym['object'])
                else:
                    error_message += 'Algorithm ' + str(algo) + ': Every model has to contain fit and predict ' \
                                                                'logic, therefore the methods `fit()` and ' \
                                                                '`predict()` have to be called at least once\n'
                    break
            else:
                # parse all parameter from training sql as well as from the predict sql
                params = self._parse_params(fit_param_sql, 'train') + self._parse_params(predict_param_sql, 'apply')

                # Parse definitions from abap type definitions with regex to find target name and type of the
                # predicted score
                # If `findall()` doesn't find exactly one match in each string an error occurred
                last_column_name_out = re.findall("[A-Za-z_]+(?= TYPE [A-Za-z0-9 ]+,\n$)", predict_output['abap_type'])
                last_column_name_in = re.findall("[A-Za-z0-9_]+(?= TYPE [A-Za-z0-9 ]+,\n$)", fit_input['abap_type'])
                first_column_name_in = re.findall("^.+(?= TYPE)", fit_input['abap_type'])
                if len(last_column_name_in) == 1 and len(first_column_name_in) == 1 \
                        and len(last_column_name_out) == 1:
                    last_column_name_in = last_column_name_in[0]
                    first_column_name_in = first_column_name_in[0]
                    last_column_name_out = last_column_name_out[0]
                else:
                    error_message += 'Error in abap definition of table input structure and prediction result.\n'
                    continue

                amdp_writer.generate(self.config.get_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH_ABAP), algo,
                                     pal_fit_sql, pal_predict_sql, params, fit_input, layer['fit']['output_tables'],
                                     predict_output, layer['predict']['output_tables'], first_column_name_in,
                                     last_column_name_out,
                                     last_column_name_in,
                                     training_dataset, apply_dataset,
                                     no_reason_features)  # parsed parameter in [(name, value, role),...] format
        if error_message != '':
            raise ValueError(error_message)

    @staticmethod
    def _parse_params(param_sql_raw, role):
        """
        Parse sql lines containing the parameter definitions. In the sql code all the parameters
        are defined by four arrays, where the first one contains the parameter name, and one of the other
        three contains the value fitting to the parameter, while the other two are NULL. This format
        should be changed into a simple key-value based storage.

        Parameters
        ----------
        param_sql_raw : List
            A list of sql statements each of them belonging to the param definition section
        role : str
            The role defines whether the sql belongs to training or applying
            only possible values are `train` and `apply`, otherwise throws

        Returns
        -------
            array of tuples, where each tuple describes a parameter like (name, value, role)

        """
        if role != "train" and role != "apply":
            raise ValueError("role value can only be 'train' or 'apply', NOT {}".format(role))
        NULL_VALUE = 'NULL'
        params = []
        param_names = []
        for i in range(0, len(param_sql_raw), 4):
            name = AMDPGenerator._parse_line(param_sql_raw[i])
            param_i = AMDPGenerator._parse_line(param_sql_raw[i + 1])
            param_d = AMDPGenerator._parse_line(param_sql_raw[i + 2])
            param_s = AMDPGenerator._parse_line(param_sql_raw[i + 3])
            if param_i == NULL_VALUE and param_d == NULL_VALUE:
                if name not in param_names:
                    params.append((name, param_s, role))
                    param_names.append(name)
                else:
                    params[param_names.index(name)] = (name, params[param_names.index(name)][1] + ',' + param_s, role)
            elif param_i == NULL_VALUE and param_s == NULL_VALUE:
                params.append((name, float(param_d), role))
                param_names.append(name)
            elif param_d == NULL_VALUE and param_s == NULL_VALUE:
                params.append((name, int(param_i), role))
                param_names.append(name)
        return params

    @staticmethod
    def _parse_line(_sql):
        """
        Parse a single line from the param definitions from sql to get the name or the value of the parameter

        Parameters
        ----------
        _sql: str
            parameter definition sql line

        Returns
        -------
        Either the name of the following parameter, or the value

        Examples
        --------
        param_name[0] := N'PARAMETER_NAME'; -> PARAMETER_NAME
        string_value[0] := N'VALUE'; -> VALUE
        double_value[0] := NULL; -> NULL
        int_value[0] := 1; -> 1
        """
        return re.findall(":= (?:N')?([0-9A-Za-z_. \\[\\],{}-]+)'?;", _sql)[0]

    @staticmethod
    def _extract_params_definition_from_sql(raw_sql):
        """
        Find the code snippet containing the parameter definition from the sql procedure

        Parameters
        ----------
        raw_sql : list
            a list of sql statements

        Returns
        -------
        List of sql statements each of them belonging to the param definition section
        """
        start_index, end_index = None, None
        for i, line in enumerate(raw_sql):
            if re.match("param_name\\[[1-9]+\\] := .+;", line) and not start_index:
                start_index = i
            if re.match("params = UNNEST(.+)", line):
                end_index = i
                break
        if start_index is None:
            start_index = end_index
        return raw_sql[start_index:end_index]

    @staticmethod
    def _find_pal_function_from_sql(raw_sql):
        """
        Extract the specific function call of the PAL function from the sql code. Nevertheless it only detects
        the synonyms that have to be resolved afterwards
        Parameters
        ----------
        raw_sql : list
            a list of sql statements

        Returns
        -------
        The procedure name synonym
        CALL "SYS_AFL.PAL_RANDOM_FORREST" (...) -> SYS_AFL.PAL_RANDOM_FORREST"
        """
        for line in raw_sql:
            calls = re.findall('CALL \"(.+)\".+,', line)
            if len(calls) > 0:
                return calls[0]
