"""
This module handles the generation of the files that represent the
artifacts. Currently this is experimental code only.
"""
#pylint: disable=line-too-long, too-many-arguments, too-many-locals, too-many-branches, too-many-statements
import os
import re
from hana_ml.artifacts.generators.sql_processor import SqlProcessorConsumptionLayer
from hana_ml.artifacts.utils import StringUtils
from .filewriter_base import FileWriterBase
from ...config import ConfigConstants

class AMDPWriter(FileWriterBase):
    """
    This class generates a amdp file
    """

    def generate(self, path, algorithm, train_procedure, predict_procedure, params, train_input_structure,
                 train_output_tbls, predict_out, predict_additional_output_tbls, key_column,
                 confidence_column, target_column, training_dataset, apply_dataset, no_reason_features):
        """
        Generate the amdp file
        """
        # Add key field to training input structure if it doesn't exist yet, due to ABAP requiring one (and exactly one)
        train_input_structure = train_input_structure['abap_type']
        has_id_index = [i for i, param in enumerate(params) if param[0] == "HAS_ID"]
        if len(has_id_index) == 1:
            if params[has_id_index[0]][1] != 1:
                params[has_id_index[0]] = ('HAS_ID', int(1), params[has_id_index[0]][2])
                train_input_structure = "        id TYPE int4,\n" + train_input_structure
                key_column = 'ID'
        fit_input_list = train_input_structure.split("\n")
        print(fit_input_list[-1])
        if fit_input_list[-1].isspace() or (fit_input_list[-1] == "") or (fit_input_list[-1] == '\n'):
            fit_input_list = fit_input_list[:-1]
        for num, item in enumerate(fit_input_list):
            fit_input_list[num] = "        " + item
        train_input_structure = "\n".join(fit_input_list)
        prediction_structure_out = predict_out['abap_type']
        predict_output_list = prediction_structure_out.split("\n")
        if predict_output_list[-1].isspace() or (predict_output_list[-1] == "") or (predict_output_list[-1] == '\n'):
            predict_output_list = predict_output_list[:-1]
        for num, item in enumerate(predict_output_list):
            predict_output_list[num] = "        " + item
        prediction_structure_out = "\n".join(predict_output_list)
        out_columns = re.findall("([A-Za-z_]+) TYPE ([^,]+)", prediction_structure_out)
        in_columns = re.findall("([A-Za-z_]+) TYPE ([^,]+)", train_input_structure)
        target_cast_type = ''
        found_class = False
        for i, col in enumerate(out_columns):
            if col[0].upper() == 'CLASS' and ' LENGTH ' in col[1].upper():
                # class column
                found_class = True
                target_type = [col for col in in_columns if col[0] == target_column][0]
                if ('UnifiedClassification' in algorithm) or ('UnifiedRegression' in algorithm):
                    prediction_structure_out = prediction_structure_out.replace(col[1], target_type[1])
                else:
                    prediction_structure_out = prediction_structure_out.replace(col[1], 'vt_cast_output' + str(i))
                target_cast_type += 'vt_cast_output{} TYPE {},\n'.format(i, target_type[1])
            elif i == len(out_columns)-2 and not found_class:
                # second last
                target_type = [col for col in in_columns if col[0] == target_column][0]
                if ('UnifiedClassification' in algorithm) or ('UnifiedRegression' in algorithm):
                    prediction_structure_out = prediction_structure_out.replace(col[1], target_type[1])
                else:
                    prediction_structure_out = prediction_structure_out.replace(col[1], 'vt_cast_output' + str(i))
                target_cast_type += 'vt_cast_output{} TYPE {},\n'.format(i, target_type[1])
            elif ' LENGTH ' in col[1].upper():
                if ('UnifiedClassification' in algorithm) or ('UnifiedRegression' in algorithm):
                    prediction_structure_out = prediction_structure_out.replace(col[1], target_type[1])
                else:
                    prediction_structure_out = prediction_structure_out.replace(col[1], 'vt_cast_output' + str(i))
                target_cast_type += 'vt_cast_output{} TYPE {},\n'.format(i, col[1])

        # Extract only the additional tables from the prediction tables (not the standard output)
        predict_additional_output_tbls = predict_additional_output_tbls[1:]
        self.write_file(path, algorithm, train_procedure, predict_procedure, params, train_input_structure,
                        train_output_tbls, prediction_structure_out, predict_additional_output_tbls,
                        key_column, confidence_column, target_column, target_cast_type,
                        training_dataset, apply_dataset, no_reason_features)

    def write_file(self, path, algorithm, train_procedure, predict_procedure, params, train_structure_in,
                   train_output_tbls, prediction_structure_out, predict_additional_output_tbls, key_column,
                   confidence_column, target_column, target_cast_type, training_dataset, apply_dataset, no_reason_features):
        """
        Write the amdp file
        """
        if ('UnifiedClassification' in algorithm) or ('UnifiedRegression' in algorithm):
            target_cast_type = ''
        filename = ConfigConstants.AMDP_TEMPLATE_FILENAME
        if 'UnifiedClassification' in algorithm:
            filename = ConfigConstants.AMDP_TEMPLATE_UNIFIED_CLASSIFICATION_FUNCTION_FILENAME
        if 'UnifiedRegression' in algorithm:
            filename = ConfigConstants.AMDP_TEMPLATE_UNIFIED_REGRESSION_FUNCTION_FILENAME
        template_file_path = os.path.join(os.path.dirname(__file__), ConfigConstants.TEMPLATE_DIR,
                                          filename)
        app_id = self.config.get_entry(ConfigConstants.CONFIG_KEY_PROJECT_NAME)

        output_sql_variable_names = []
        data_definitions_for_output_types = []
        method_signature = []
        procedure_parameters = []
        metric_map = {
            'VAR_IMPORTANCE': 'imp',
            # No value for displaying out_of_bag_error
            # 'OOB_ERR': 'out_of_bag',
            'CM': 'confusion',
            'GI': 'gen',
            'METRICS': 'metrics',
            # 'STATS', 'CV', 'STATISTICS : NONE
        }
        special_dtype_map = {
            'VAR_IMPORTANCE': 'shemi_variable_importance_t',
            'CM': 'shemi_confusion_matrix_t'
        }
        for i, output_table in enumerate(train_output_tbls):
            if output_table['cat'] is not None:
                output_table['name'] = "et_{}{}".format(output_table['cat'][:15], i)
            if output_table['name'] in output_sql_variable_names:
                continue
            # Append variable name to be used in the sql call afterwards
            output_sql_variable_names.append('et_model' if output_table['cat'] == 'MODEL' else output_table['name'])

            dtype = special_dtype_map.get(output_table['cat'])
            if dtype is None and (('UnifiedClassification' not in algorithm) or ('UnifiedRegression' not in algorithm) or output_table['cat'] == "MODEL"):
                abap_structure = SqlProcessorConsumptionLayer(self.config).build_abap_datatype(
                    output_table['table_type'])[:-1]
                if output_table['cat'] == "MODEL":
                    # Only in the case of the model structure being defined, use string instead of NCHAR
                    abap_structure = abap_structure.replace('NCHAR', 'string')
                    model_structure = 'tt_{}'.format(output_table['name'])
                # Generate abap structure type and table type for every table
                data_definitions_for_output_types.append(
                    """
                    BEGIN OF st_{},
                    {}
                    END OF st_{},
                    tt_{} TYPE STANDARD TABLE OF st_{} WITH DEFAULT KEY""".format(
                        output_table['name'],
                        abap_structure,
                        output_table['name'],
                        output_table['name'],
                        output_table['name'])
                )
                dtype = 'tt_{}'.format(output_table['name'])

            # Generate EXPORTING method signature of training
            param_name = 'et_model' if output_table['cat'] == 'MODEL' else output_table['name']
            method_signature.append("VALUE({}) TYPE {}".format(param_name, dtype))

            # Add procedure parameter metadata for definition
            if output_table['cat'] == 'MODEL':
                procedure_parameters.append("( name = 'et_model' role = cl_hemi_constants=>cs_proc_role-model )")
            else:
                additional_info = metric_map.get(output_table['cat'])
                if additional_info is not None:
                    additional_info = "add_info = '{}' ".format(additional_info)
                else:
                    additional_info = ''
                procedure_parameters.append(
                    "( name = '{}' role = cl_hemi_constants=>cs_proc_role-stats {})".format(
                        output_table['name'], additional_info))

        # iterate over all fields with the corresponding data type (field_name, dtype)
        # to cast the prediction result accordingly
        select_result_colums = []
        for field_name, dtype in re.findall("([A-Za-z_]+) TYPE ([^ ,]+)", prediction_structure_out):
            dtype = dtype.upper()
            # add type cast
            # if the score type is a primitive data type, ensure that you pick the correct cast
            if dtype in ConfigConstants.ABAP_PRIMITIVE_DATATYPE_CASTS.keys():
                dtype = ConfigConstants.ABAP_PRIMITIVE_DATATYPE_CASTS.get(dtype)
            sql_cast = "cast(result.{0} as \"$ABAP.type( {1} )\") as {0}".format(field_name, dtype)
            select_result_colums.append(sql_cast)

        # Create filename `Z_CL_` as naming-convention
        amdp_name = 'Z_CL_' + app_id.upper() + '_' + self.config.get_entry(ConfigConstants.CONFIG_KEY_VERSION)

        # Add additional predict outputs
        add_predict_output = ',' + ','.join([tbl['name'] for tbl in predict_additional_output_tbls])
        add_predict_output = '' if add_predict_output == ',' else add_predict_output

        # Initiate the each table definition with `Value #( `
        model_params = ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_INIT
        model_params_default = ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_INIT

        # Add all parameter with their default values
        key_field_description = ''
        for _name, _value, _role in params:
            _type = 'integer' if isinstance(_value, int) else 'double' if isinstance(_value, float) else 'string'
            if _name == 'HAS_ID' and _value == 1:
                key_field_description = ConfigConstants.AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION.replace(
                    ConfigConstants.AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION_ID, key_column
                )
            model_params += StringUtils.multi_replace(
                ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE,
                {
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_NAME: _name,
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_ROLE: _role,
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_TYPE: _type,
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_CONFIGURABLE:
                        ConfigConstants.AMDP_TEMPLATE_ABAP_TRUE,  # Only ABAP_FALSE/ABAP_TRUE possible
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_CONTEXT:
                        ConfigConstants.AMDP_TEMPLATE_ABAP_FALSE  # Only ABAP_FALSE/ABAP_TRUE possible
                }
            )
            if _name not in model_params_default:
                model_params_default += StringUtils.multi_replace(
                    ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE,
                    {
                        ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE_NAME: _name,
                        # _value can be string, int or float, therefore cast it to str
                        ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE_VALUE: str(_value)
                    })
        # Wrap up the value definitions with ` )`
        model_params += ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_END
        model_params_default += ConfigConstants.AMDP_TEMPLATE_MODEL_PARAM_END
        file_name = amdp_name + ConfigConstants.AMDP_FILE_EXTENSION
        reason_code_field = ''
        result_reason_code_fields = ''
        if 'UnifiedClassification' in algorithm:
            for item in range(1, no_reason_features + 1):
                reason_code_field += ' ' * 8
                reason_code_field += """reason_code_feature_{0}    TYPE shemi_reason_code_feature_name,
            reason_code_percentage_{0} TYPE shemi_reason_code_feature_pct,""".format(item)
                result_reason_code_fields += ' ' * 23
                result_reason_code_fields += """trim(both '"' from json_query(result.reason_code, '$[{0}].attr')) as reason_code_feature_{1},
                        json_query(result.reason_code, '$[{0}].pct' ) as reason_code_percentage_{1}""".format(item - 1, item)
                if item < no_reason_features:
                    reason_code_field += "\n"
                    result_reason_code_fields += ",\n"
        train_structure_in_m = train_structure_in.lower().replace(" type ", " TYPE ")
        select_predict_colums = []
        for field_name in re.findall("([A-Za-z_]+) TYPE", train_structure_in_m):
            if field_name != target_column.lower():
                select_predict_colums.append(field_name)

        replacements = {
            ConfigConstants.AMDP_TEMPLATE_AMDP_NAME_PLACEHOLDER: amdp_name.lower(),
            ConfigConstants.AMDP_TEMPLATE_TRAIN_INPUT_STRUCTURE: train_structure_in_m,
            ConfigConstants.AMDP_TEMPLATE_INPUT_COLUMNS_WITHOUT_KEY: (",".join(
                re.sub("(TYPE [^,]+)|\n", '', train_structure_in).split(',')[1:-1])).replace(" ", "").replace(",", ", "),

            # Prediction structure not custom defined, as it's always the exact same as coming from pal itself unless
            # you remodel it inside the `predict_with_model_version` inside the abap class after the pal call. But this
            # isn't yet possible from inside the python code:
            ConfigConstants.AMDP_TEMPLATE_PREDICTION_STRUCTURE: prediction_structure_out.lower().replace(" type ", " TYPE "),
            ConfigConstants.AMDP_TEMPLATE_MODEL_STRUCTURE: model_structure,

            ConfigConstants.AMDP_TEMPLATE_RESULT_FIELDS: ("\n" + " " * 23 + ", ").join(select_result_colums),

            # Model structure is defined by the base class, the generated abap class is inheriting from anyways.
            # Therefore do not hand over the structure from the sql as the abap structure is correctly and individually
            # designed for every classifier/regressor in the base class
            # ConfigConstants.AMDP_TEMPLATE_SAVED_MODEL_STRUCTURE: model_structure_out,

            # Don't insert the used Training/prediction dataset as they might not be CDS view and pal can only handle
            # cds views
            # ConfigConstants.AMDP_TEMPLATE_ES_META_DATA_DEFINITION_TRAIN_SET + fit_input['sql_vars'][0] + '\'.',
            # ConfigConstants.AMDP_TEMPLATE_ES_META_DATA_DEFINITION_PREDICT_SET + predict_input['sql_vars'][0] + '\'.',

            ConfigConstants.AMDP_TEMPLATE_PARAMETER: model_params,
            ConfigConstants.AMDP_TEMPLATE_ADDITIONAL_PREDICT_OUTPUT: add_predict_output,
            ConfigConstants.AMDP_TEMPLATE_CONFIDENCE_NAME: confidence_column,

            ConfigConstants.AMDP_TEMPLATE_SQL_OUTPUT_VARIABLES: ','.join(output_sql_variable_names),
            ConfigConstants.AMDP_TEMPLATE_TRAIN_OUTPUT_TYPE_DEFINITIONS: ',\n'.join(data_definitions_for_output_types),
            ConfigConstants.AMDP_TEMPLATE_TRAIN_METHOD_SIGNATURE: '\n'.join(method_signature),
            ConfigConstants.AMDP_TEMPLATE_PROCEDURE_PARAMETERS: '\n'.join(procedure_parameters),

            ConfigConstants.AMDP_TEMPLATE_PAL_TRAIN_CALL: train_procedure.lower(),
            ConfigConstants.AMDP_TEMPLATE_PAL_PREDICT_CALL: predict_procedure.lower(),

            ConfigConstants.AMDP_TEMPLATE_PARAMETER_DEFAULT: model_params_default,
            ConfigConstants.AMDP_TEMPLATE_TARGET_COLUMN: target_column,
            ConfigConstants.AMDP_TEMPLATE_OUTPUT_CAST_TYPE: target_cast_type,
            ConfigConstants.AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION_PLACEHOLDER: key_field_description,
            ConfigConstants.AMDP_TEMPLATE_TRAINING_DATASET: training_dataset,
            ConfigConstants.AMDP_TEMPLATE_APPLY_DATASET: apply_dataset,
            ConfigConstants.AMDP_TEMPLATE_REASON_CODE: reason_code_field,
            ConfigConstants.AMDP_TEMPLATE_RESULT_REASON_CODE_FIELDS: result_reason_code_fields,
            ConfigConstants.AMDP_TEMPALTE_PREDICT_DATA_COLS: ", ".join(select_predict_colums)

        }

        self.write_template(path, file_name, template_file_path, replacements)
