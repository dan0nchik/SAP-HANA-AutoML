"""
This module allows allows interaction with the configuration functionality.

It consists of two parts.
1. The ConfigHandler which provides helper functions to support easy setting
and retrieving of configurations.
2. Config Constants for easy access to config values.
"""
#pylint: disable=line-too-long, too-few-public-methods, too-many-arguments
import os
import json
from hana_ml.artifacts.utils import StringUtils


class ConfigConstants(object):
    """
    This class provides a central repository of config constants they are defined here. This allows
    for easy adjustments without needed to change all the literals in the code.
    """
    # ---------------------------------------
    # ---- Generic Config constants ----
    # ---------------------------------------

    # Generation Types for Consumption Layer
    GENERATION_MERGE_NONE = 1  # Each step (ie partition or fit) is on its own. For example a procedure for partition and one for fit.
    GENERATION_MERGE_PARTITION = 2  # Merges the partition into the related step fit or predict. So not to create unnecessary data copies. All other steps will stay isolated.
    GENERATION_GROUP_NONE = 11  # No grouping is applied. This means that solution specific implementation will define how to deal with this
    GENERATION_GROUP_FUNCTIONAL = 12  # Grouping is based on functional grouping. Meaning that logical related elements such as partiion / fit / and related score will be put together
    GROUP_FIT_TYPE = 'fit'
    GROUP_PREDICT_TYPE = 'predict'
    GROUP_FIT_PREDICT_TYPE = 'fit_predict'
    GROUP_UKNOWN_TYPE = 'unknown'
    GROUP_IDENTIFIER_MERGE_ALL = 'all_merged'

    # Generic Config Keys
    CONFIG_KEY_OUTPUT_DIR = 'outputdir'
    CONFIG_KEY_OUTPUT_PATH = 'outputpath'
    CONFIG_KEY_PROJECT_NAME = 'project_name'
    CONFIG_KEY_VERSION = 'version'
    CONFIG_KEY_MERGE_STRATEGY = 'merge_type'
    CONFIG_KEY_GROUP_STRATEGY = 'group_type'

    # Config Data
    DATA_CONVERSION_HDBTABLE_HDBDD_FILE = 'hdbtable_to_hdbdd_datatype_mapping.json'
    DATA_CONVERSION_HDBTABLE_ABAP_FILE = 'hdbtable_to_abap_datatype_mapping.json'
    DATA_CATEGORY_IDENTIFIERS_FILE = 'category_identifiers.json'
    CONFIG_DATA_DIR = 'data'
    CONFIG_KEY_HDBTABLE_TO_HDBDD_TYPES = 'hdbtable_to_hdbdd_datatypes'
    CONFIG_KEY_HDBTABLE_TO_ABAP_TYPES = 'hdbtable_to_abap_datatypes'
    CONFIG_KEY_CATEGORY_IDENTIFIERS = 'category_identfiers'
    CONFIG_KEY_CATEGORY_IDENTIFIER_MODEL = 'category_identfier_model'
    CONFIG_KEY_CATEGORY_IDENTIFIER_METRIC = 'category_identfier_metric'
    CONFIG_KEY_CATEGORY_IDENTIFIER_FITTED = 'category_identfier_fitted'
    # ---------------------------------------
    # ---- HANA Related Config constants ----
    # ---------------------------------------

    # HANA Related Config Keys
    CONFIG_KEY_APPID = 'appid'
    CONFIG_KEY_OUTPUT_PATH_HANA = 'output_path_hana'
    CONFIG_KEY_MODULE_NAME = 'module_name'
    CONFIG_KEY_SCHEMA = 'schema'
    CONFIG_KEY_GRANT_SERVICE = 'ups'

    CONFIG_KEY_SQL_PROCESSED = 'sql_processed'

    CONFIG_KEY_CDS_CONTEXT = 'cds_context'

    CONFIG_KEY_OUTPUT_PATH_MODULE = 'output_path_module'
    CONFIG_KEY_OUTPUT_PATH_MODULE_SRC = 'output_path_module_src'
    CONFIG_KEY_MODULE_TEMPLATE_PATH = 'module_template_path'
    CONFIG_KEY_GRANTS_PATH = 'grants_path'
    CONFIG_KEY_SYNONYMS_PATH = 'synonyms_path'
    CONFIG_KEY_PROCEDURES_PATH = 'procedures_path'
    CONFIG_KEY_AMDP_PATH = 'amdp_path'
    CONFIG_KEY_ROLES_PATH = 'roles_path'
    CONFIG_KEY_CDS_PATH = 'cds_path'

    # --SDA
    CONFIG_KEY_SDA_APPID = CONFIG_KEY_APPID + '_sda'
    CONFIG_KEY_SDA_OUTPUT_PATH_HANA = CONFIG_KEY_OUTPUT_PATH_HANA + '_sda'
    CONFIG_KEY_SDA_MODULE_NAME = CONFIG_KEY_MODULE_NAME + '_sda'
    CONFIG_KEY_SDA_SCHEMA = CONFIG_KEY_SCHEMA + '_sda'
    CONFIG_KEY_SDA_GRANT_SERVICE = CONFIG_KEY_GRANT_SERVICE + '_sda'
    CONFIG_KEY_SDA_REMOTE_SOURCE = 'remote_source_sda'
    CONFIG_KEY_SDA_OUTPUT_PATH_MODULE = CONFIG_KEY_OUTPUT_PATH_MODULE + '_sda'
    CONFIG_KEY_SDA_OUTPUT_PATH_MODULE_SRC = CONFIG_KEY_OUTPUT_PATH_MODULE_SRC + '_sda'
    CONFIG_KEY_SDA_GRANTS_PATH = CONFIG_KEY_GRANTS_PATH + '_sda'
    CONFIG_KEY_SDA_SYNONYMS_PATH = CONFIG_KEY_SYNONYMS_PATH + '_sda'
    CONFIG_KEY_SDA_PROCEDURES_PATH = CONFIG_KEY_PROCEDURES_PATH + '_sda'
    CONFIG_KEY_SDA_AMDP_PATH = 'amdp_path' + '_sda'
    CONFIG_KEY_SDA_ROLES_PATH = CONFIG_KEY_ROLES_PATH + '_sda'
    CONFIG_KEY_SDA_CDS_PATH = CONFIG_KEY_CDS_PATH + '_sda'
    CONFIG_KEY_SDA_VIRTUALTABLE_PATH = 'virtual_table_sda'
    CONFIG_KEY_SDA_MODEL_ONLY = 'model_only'
    CONFIG_KEY_DATA_SOURCE_MAPPING = 'data_source_mapping'

    # Project locations
    HANA_BASE_PATH = 'hana'
    SDA_HANA_BASE_PATH = 'hana_sda'
    PROJECT_TEMPLATEDR = 'templates'
    PROJECT_TEMPLATE_BASE_DIR = 'hana'
    PROJECT_TEMPLATE_BASE_STRUCT = 'base_structure'
    PROJECT_TEMPLATE_BASE_SDA_STRUCT = 'base_sda_structure'
    MODULE_SOURCE_PATH = 'src'
    GRANTS_SOURCE_PATH = 'grants'
    SYNONYMS_SOURCE_PATH = 'synonyms'
    PROCEDURES_SOURCE_PATH = 'procedures'
    AMDP_SOURCE_PATH = 'amdps'
    ROLES_SOURCE_PATH = 'roles'
    VIRTUAL_TABLE_SOURCE_PATH = 'virtualtable'
    CDS_SOURCE_PATH = 'cds'

    # FileWriter - Generic
    TEMPLATE_DIR = 'templates'

    # FileWriter - MTAYamlWriter
    YAML_FILE_NAME = 'mta.yaml'
    YAML_TEMPLATE_FILE = 'tmp.yaml'
    YAML_TEMPLATE_APPID_PLACEHOLDER = '<<MODULE_APPID>>'
    YAML_TEMPLATE_VERSION_PLACEHOLDER = '<<MODULE_VERSION>>'
    YAML_TEMPLATE_NAME_PLACEHOLDER = '<<MODULE_NAME>>'
    YAML_TEMPLATE_SCHEMA_PLACEHOLDER = '<<SCHEMA_NAME>>'
    YAML_TEMPLATE_UPS_GRANT_PLACEHOLDER = '<<USP_FOR_GRANTS>>'

    # FileWriter - HDBGrantWriter
    GRANT_FILE_NAME = "grants"
    GRANT_TMP_FILE_NAME = 'tmp.hdbgrant'
    GRANT_TMP_SDA_FILE_NAME = 'tmp_sda.hdbgrant'
    GRANT_FILE_EXTENSION = '.hdbgrants'
    GRANT_TEMPLATE_SDA_REMOTE_SOURCE = '<<REMOTE_SOURCE>>'
    GRANT_TEMPLATE_SCHEMA_PRIVILEGES = '<<SCHEMA_PRIVILEGES>>'
    GRANT_TEMPLATE_SCHEMA_PRIVILEGES_WITH = '<<SCHEMA_PRIVILEGES_WITH>>'

    # FileWriter - HDBProcedureWriter
    PROCEDURE_FILE_EXTENSION = '.hdbprocedure'
    PROCEDURE_TEMPLATE_FILE = 'tmp.hdbprocedure'
    PROCEDURE_TEMPLATE_SQL_PLACEHOLDER = '<<SQL_PLACEHOLDER>>'
    PROCEDURE_TEMPLATE_PROC_INTERFACE = '<<INTERFACE>>'
    PROCEDURE_TEMPLATE_PROC_NAME = '<<PROCEDURE_NAME>>'

    # FileWriter - HDBSynonymWriter
    SYNONYM_FILE_NAME = 'synonyms'
    SYNONYM_FILE_EXTENSION = '.hdbsynonym'

    # FileWriter - HDBRole
    ROLE_FILE_EXTENSION = '.hdbrole'
    ROLE_TEMPLATE_FILENAME = 'tmp.hdbrole'
    ROLE_TEMPLATE_FILENAME_WITH = 'tmp_with.hdbrole'
    ROLE_TEMPLATE_ROLE_NAME_PLACEHOLDER = '<<ROLE_NAME>>'

    # FileWriter - HDBVirtualTable
    VIRTUAL_TABLE_FILE_EXTENSION = '.hdbvirtualtable'
    VIRTUAL_TABLE_TEMPLATE_FILENAME = 'tmp.hdbvirtualtable'
    VIRTUAL_TABLE_TEMPLATE_CONNECTION_NAME_PLACEHOLDER = '<<CONNECTION_NAME>>'
    VIRTUAL_TABLE_TEMPLATE_VIRTUAL_TABLE_PLACEHOLDER = '<<VIRTUAL_TABLE>>'
    VIRTUAL_TABLE_TEMPLATE_SOURCE_SCHEMA_TABLE_PLACEHOLDER = '<<SOURCE_SCHEMA_TABLE>>'

    # FileWriter - HDBCDS
    CDS_FILE_EXTENSION = '.hdbcds'
    CDS_TEMPLATE_FILENAME = 'tmp.hdbcds'
    CDS_TEMPLATE_CONTEXT_NAME = '<<CONTEXT_NAME>>'
    CDS_TEMPLATE_CONTEXT_CONTENT = '<<CONTEXT_CONTENT>>'

    # ---------------------------------------
    # ---- ABAP Related Config constants ----
    # ---------------------------------------

    # ABAP Related Config Keys
    CONFIG_KEY_OUTPUT_PATH_ABAP = 'output_path_abap'
    ABAP_BASE_PATH = 'abap'

    # FileWriter - AMDP
    AMDP_FILE_EXTENSION = '.abap'
    AMDP_SUPPORTED_ALGORITHMS = ['randomforest', 'naivebayes', 'logistic', 'mlp', 'decisiontree',
                                 'hybridgradientboosting', 'sv', 'unifiedclassification', 'unifiedregression']
    AMDP_TEMPLATE_FILENAME = 'tmp_hemi.abap'
    AMDP_TEMPLATE_UNIFIED_CLASSIFICATION_FUNCTION_FILENAME = 'tmp_hemi_unified_classification_func.abap'
    AMDP_TEMPLATE_UNIFIED_REGRESSION_FUNCTION_FILENAME = 'tmp_hemi_unified_regression_func.abap'
    AMDP_TEMPLATE_AMDP_NAME_PLACEHOLDER = '<<AMDP_NAME>>'
    AMDP_TEMPLATE_TRAIN_INPUT_STRUCTURE = '<<TRAIN_INPUT_STRUCTURE>>'
    AMDP_TEMPLATE_INPUT_COLUMNS_WITHOUT_KEY = '<<COLUMNS_WITHOUT_KEY>>'
    AMDP_TEMPLATE_PREDICTION_STRUCTURE = '<<RESULT_OUTPUT_STRUCTURE>>'
    AMDP_TEMPLATE_RESULT_FIELDS = '<<RESULT_FIELDS>>'
    AMDP_TEMPLATE_MODEL_STRUCTURE = '<<MODEL_STRUCTURE>>'
    AMDP_TEMPLATE_TRAIN_OUTPUT_TYPE_DEFINITIONS = '<<TRAIN_OUTPUT_TYPE_DEFINITIONS>>'
    AMDP_TEMPLATE_TRAIN_METHOD_SIGNATURE = '<<TRAIN_METHOD_SIGNATURE>>'
    AMDP_TEMPLATE_SQL_OUTPUT_VARIABLES = '<<SQL_OUTPUT_VARIABLES>>'
    AMDP_TEMPLATE_PROCEDURE_PARAMETERS = '<<PROCEDURE_PARAMETERS>>'
    AMDP_TEMPLATE_ADDITIONAL_PREDICT_OUTPUT = '<<ADDITIONAL_PREDICT_OUTPUTS>>'
    AMDP_TEMPLATE_CONFIDENCE_NAME = '<<CONFIDENCE_NAME>>'
    AMDP_TEMPLATE_OUTPUT_CAST_TYPE = '<<CAST_TARGET_OUTPUT>>'
    AMDP_TEMPLATE_TRAINING_DATASET = '<<TRAINING_DATASET>>'
    AMDP_TEMPLATE_APPLY_DATASET = '<<APPLY_DATASET>>'
    AMDP_TEMPLATE_REASON_CODE = '<<REASON_CODE_STRUCTURE>>'
    AMDP_TEMPLATE_RESULT_REASON_CODE_FIELDS = '<<RESULT_REASON_CODE_FIELDS>>'
    AMDP_TEMPALTE_PREDICT_DATA_COLS = '<<PREDICT_DATA_COLS>>'

    ABAP_PRIMITIVE_DATATYPE_CASTS = {
        'I': 'INT4',
        'D': 'DATS',
        'S': 'int2',
        'B': 'int1',
        'F': 'FLTP',
        'T': 'TIMS',
    }

    AMDP_TEMPLATE_TARGET_COLUMN = '<<TARGET_COLUMN>>'
    AMDP_TEMPLATE_PARAMETER = '<<PARAMETER>>'
    AMDP_TEMPLATE_PARAMETER_DEFAULT = '<<PARAMETER_DEFAULT>>'
    AMDP_TEMPLATE_PAL_TRAIN_CALL = '<<PAL_TRAIN_CALL>>'
    AMDP_TEMPLATE_PAL_PREDICT_CALL = '<<PAL_PREDICT_CALL>>'

    AMDP_TEMPLATE_MODEL_PARAM_INIT = "VALUE #( "
    AMDP_TEMPLATE_MODEL_PARAM_END = ")"
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE = "( name = '<<NAME>>' type = cl_hemi_constants=>cs_param_type-<<TYPE>> role = " \
                                       "cl_hemi_constants=>cs_param_role-<<ROLE>> configurable = <<CONFIGURABLE>> " \
                                       "has_context = <<CONTEXT>> )\n      "
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_NAME = "<<NAME>>"
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_ROLE = "<<ROLE>>"
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_TYPE = "<<TYPE>>"
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_CONFIGURABLE = "<<CONFIGURABLE>>"
    AMDP_TEMPLATE_MODEL_PARAM_SAMPLE_CONTEXT = "<<CONTEXT>>"
    AMDP_TEMPLATE_ABAP_FALSE = "abap_false"
    AMDP_TEMPLATE_ABAP_TRUE = "abap_true"

    AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION_PLACEHOLDER = "<<KEY_FIELD_DESCRIPTION>>"
    AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION = " ( name = '<<ID>>' role = cl_hemi_constants=>cs_field_role-key )"
    AMDP_TEMPLATE_KEY_FIELD_DESCRIPTION_ID = "<<ID>>"

    AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE = "( name = '<<NAME>>' value = '<<VALUE>>' )\n      "
    AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE_NAME = "<<NAME>>"
    AMDP_TEMPLATE_MODEL_PARAM_DEFAULT_SAMPLE_VALUE = "<<VALUE>>"

    AMDP_TEMPLATE_MULTIPLE_PARAMETERS_PARSING = '<<MULTIPLE_PARAMETER_PARSING>>'
    # ------------------------------------------
    # ---- DataHub Related Config constants ----
    # ------------------------------------------
    # Graph Generation Scope
    GRAPH_GENERATION_TYPE_SEPERATE = 1  # Depends on generation type but keeps the consumption layer elements seperate in different graphs
    GRAPH_GENERATION_TYPE_COMBINED = 2  # Combines all consumption layer elements in one graph seperated by a operator per consumption layer element

    # Location Constants
    DATAHUB_BASE_PATH = 'datahub'
    GRAPH_FILE_NAME = 'graph'
    GRAPH_FILE_EXTENSION = '.json'

    # Generic Constants
    DATAHUB_HANAML_OPERATOR_COMPONENT = 'contrib.hanaml.python_api'
    DATAHUB_MLAPI_OPERATOR_MDL_PROD_COMPONENT = 'com.sap.ml.artifacts.producer'
    DATAHUB_MLAPI_OPERATOR_MDL_CONS_COMPONENT = 'com.sap.ml.artifacts.consumer'
    DATAHUB_MLAPI_OPERATOR_METRICS_COMPONENT = 'com.sap.ml.submitMetrics'
    DATAHUB_OPERATOR_CONVERTER_TOBLOB_COMPONENT = 'com.sap.util.toBlobConverter'
    DATAHUB_OPERATOR_CONVERTER_TOSTRING_COMPONENT = 'com.sap.util.toStringConverter'
    DATAHUB_OPERATOR_CONVERTER_TOMESSAGE_COMPONENT = 'com.sap.util.toMessageConverter'
    DATAHUB_OPERATOR_GRAPH_TERMINATOR_COMPONENT = 'com.sap.util.graphTerminator'
    DATAHUB_OPERATOR_RESTAPI_COMPONENT = 'com.sap.openapi.server'
    DATAHUB_OPERATOR_RESPONSE_INTER_COMPONENT = 'com.sap.util.responseinterceptor'
    DATAHUB_OPERATOR_CONSTANT_GENERATOR_COMPONENT = 'com.sap.util.constantGenerator'

    # Operator Specific constants
    # Model Producer
    DATAHUB_OPERATOR_MDL_PROD_IN_PORT_BLOB = 'inArtifact'
    DATAHUB_OPERATOR_MDL_PROD_OUT_PORT_MSG = 'outArtifact'
    # Model Consumer
    DATAHUB_OPERATOR_MDL_CONS_IN_PORT_STRING = 'inArtifactID'
    DATAHUB_OPERATOR_MDL_CONS_OUT_PORT_BLOB = 'outArtifact'
    # Metrics
    DATAHUB_OPERATOR_METRICS_IN_PORT_METRICS = 'metrics'
    DATAHUB_OPERATOR_METRICS_OUT_PORT_RESPONSE = 'response'
    # ToBlob Converter
    DATAHUB_OPERATOR_TOBLOB_IN_PORT_STRING = 'ininterface'
    DATAHUB_OPERATOR_TOBLOB_OUT_PORT_BLOB = 'outbytearray'
    # ToString Converter
    DATAHUB_OPERATOR_TOSTRING_IN_PORT_BLOB = 'ininterface'
    DATAHUB_OPERATOR_TOSTRING_IN_PORT_MESSAGE = 'inmessage'
    DATAHUB_OPERATOR_TOSTRING_OUT_PORT_STRING = 'outstring'
    # ToMessage Converter
    DATAHUB_OPERATOR_TOMESSAGE_IN_PORT_STRING = 'instring'
    DATAHUB_OPERATOR_TOMESSAGE_IN_PORT_ANY = 'inbody'
    DATAHUB_OPERATOR_TOMESSAGE_OUT_PORT_MESSAGE = 'out'
    # Graph Terminator
    DATAHUB_OPERATOR_GRAPH_TERMINATOR_IN_PORT_ANY = 'stop'
    # Python Terminator Check
    DATAHUB_OPERATOR_TERMINATOR_IN_PORT_MESSAGE = 'metrics'
    DATAHUB_OPERATOR_TERMINATOR_IN_PORT_STRING = 'artifacts'
    DATAHUB_OPERATOR_TERMINATOR_OUT_PORT_MESSAGE = 'output'
    # Constants Generator
    DATAHUB_OPERATOR_CONSTANT_GENERATOR_OUT_PORT_STRING = 'out'
    # -- Rest Enpoint Operators --
    # Python Result Client
    DATAHUB_OPERATOR_RESULT_IN_PORT_MESSAGE = 'restrequestin'
    DATAHUB_OPERATOR_RESULT_OUT_PORT_MESSAGE = 'restrequestout'
    DATAHUB_OPERATOR_RESULT_IN_PORT_STRING = 'datain'
    # Response Interceptor
    DATAHUB_OPERATOR_RESPONSE_INTER_REQUEST_IN_PORT_MESSAGE = 'in'
    DATAHUB_OPERATOR_RESPONSE_INTER_RESPONSE_IN_PORT_MESSAGE = 'resp'
    DATAHUB_OPERATOR_RESPONSE_INTER_OUT_PORT_MESSAGE = 'out'
    # OpenAPI Servflow
    DATAHUB_OPERATOR_RESTAPI_OUT_PORT_MESSAGE = 'out'
    # DataHub Related Config Keys
    CONFIG_KEY_OUTPUT_PATH_DATAHUB = 'output_path_datahub'

    # -----------------------------------------------
    # ---- CloudFoundry Related Config constants ----
    # -----------------------------------------------
    CONFIG_KEY_OUTPUT_PATH_CF = 'cloudfoundry'


class ConfigHandler(object):
    """
    This class handles config related functions and helper functions. The main functions are:

    * Generic setting and retrieving of config values
    * Data source mapping setting and retrieving. Data source mapping can be set by the user
    * Data type mapping between different generation targets based on predefined type mapping:
        * HANA <--> ABAP data types
        * HANA <--> CDS entity data types
    * SQL Trace Element category identification based on predifined mapping between hana_ml package
    generation and what is required for artifacts generation

    For easy editing of configuration data some of the config functions have an underlying
    data json file located in the data subfolder that is loaded as part of the initialization
    of instance of this class.

    Specifically:

    * Data type mapping
    * Category identification

    """

    def __init__(self):
        """
        Class for interacting with config data.
        """
        self.config = {}
        self._process_config_data()

    def add_entry(self, key, value):
        """
        Add data to the config under the provided key

        Parameters
        ----------
        key: str
            The key under which the value needs to be stored
        value: object
            Object that will be stored under the provided key
        """
        self.config[key] = value

    def get_entry(self, key):
        """
        Get data from the config using the provided key

        Parameters
        ----------
        key: str
            The key under which the value needs to be stored

        Returns
        -------
        value: object
            Object that will be stored under the provided key
        """
        return self.config[key]

    def data_source_mapping(self, replace_str):
        """
        Apply the data source mapping on the provided input string (replace_str)

        Parameters
        ----------
        replace_str: str
            The string on which the data source mapping needs to be applied

        Returns
        -------
        transformed_str: str
            The transformed value based on the configured data source mapping

        """
        data_source_mapping = self.get_entry(ConfigConstants.CONFIG_KEY_DATA_SOURCE_MAPPING)
        return StringUtils.multi_replace(replace_str, data_source_mapping)

    def is_model_category(self, category):
        """
        Check if the category is the model category.

        Parameters
        ----------
        category: str
            The category to check

        Returns
        -------
        validated: boolean
            Whether we are dealing with a model category

        """
        return self._is_category(category, ConfigConstants.CONFIG_KEY_CATEGORY_IDENTIFIER_MODEL)

    def is_metric_category(self, category):
        """
        Check if the category is the metric category.

        Parameters
        ----------
        category: str
            The category to check

        Returns
        -------
        validated: boolean
            Whether we are dealing with a metric category

        """
        return self._is_category(category, ConfigConstants.CONFIG_KEY_CATEGORY_IDENTIFIER_METRIC)

    def is_fitted_category(self, category):
        """
        Check if the category is the fitted category.

        Parameters
        ----------
        category: str
            The category to check

        Returns
        -------
        validated: boolean
            Whether we are dealing with a fitted category

        """
        return self._is_category(category, ConfigConstants.CONFIG_KEY_CATEGORY_IDENTIFIER_FITTED)

    def _is_category(self, category, category_identfier):
        """
        Generic category validation method

        Parameters
        ----------
        category: str
            The category to check
        category: str
            The category identifier type to check (ie FITTED, MODEL)

        Returns
        -------
        validated: boolean
            Whether we are dealing with the category to validate

        """
        category_identfiers = self.get_entry(ConfigConstants.CONFIG_KEY_CATEGORY_IDENTIFIERS)
        if category and category_identfier in category_identfiers:
            if any(identifier in category for identifier in category_identfiers[category_identfier]):
                return True
        return False

    def _process_config_data(self):
        """
        Helper method to load the configuration data stored in the data folder
        """
        # HDBTABLE to HDBDD Data Type conversion replacements
        data_type_conversion_file = os.path.join(os.path.dirname(__file__), ConfigConstants.CONFIG_DATA_DIR,
                                                 ConfigConstants.DATA_CONVERSION_HDBTABLE_HDBDD_FILE)
        input_file = open(data_type_conversion_file)
        data_type_replacements = json.load(input_file)
        input_file.close()
        self.add_entry(ConfigConstants.CONFIG_KEY_HDBTABLE_TO_HDBDD_TYPES, data_type_replacements)
        # HDBTABLE to ABAP Data Type conversion replacements
        data_type_conversion_file_abap = os.path.join(os.path.dirname(__file__), ConfigConstants.CONFIG_DATA_DIR,
                                                      ConfigConstants.DATA_CONVERSION_HDBTABLE_ABAP_FILE)
        input_file = open(data_type_conversion_file_abap)
        data_type_replacements_abap = json.load(input_file)
        input_file.close()
        self.add_entry(ConfigConstants.CONFIG_KEY_HDBTABLE_TO_ABAP_TYPES, data_type_replacements_abap)
        # Category Identifiers
        category_identifiers = os.path.join(os.path.dirname(__file__), ConfigConstants.CONFIG_DATA_DIR,
                                            ConfigConstants.DATA_CATEGORY_IDENTIFIERS_FILE)
        input_file = open(category_identifiers)
        category_identifiers_dict = json.load(input_file)
        input_file.close()
        self.add_entry(ConfigConstants.CONFIG_KEY_CATEGORY_IDENTIFIERS, category_identifiers_dict)

    @staticmethod
    def init_config(project_name,
                    version,
                    grant_service,
                    outputdir,
                    generation_merge_type=ConfigConstants.GENERATION_MERGE_NONE,
                    generation_group_type=ConfigConstants.GENERATION_GROUP_FUNCTIONAL,
                    sda_grant_service=None,
                    remote_source=''):
        """
        Method to initiate the configuration.

        Parameters
        ----------
        project_name : str
            The project name which will be used across the artifacts generation such as folder that
            is created where the generated artifacts are placed.
        version : str
            The version to add to distinguish between multiple runs of the same project.
        grant_service : str
            The Cloud Foundry grant service that is used to grant the HDI container tech user the
            proper access during the deployers
        outputdir : str
            The location where the artifacts need to placed after generation.
        generation_merge_type : int
            Merge type is which operations should be merged together. There are at this stage
            only 2 options
            1: GENERATION_MERGE_NONE: All operations are generated seperately (ie. individual
            procedures in HANA)
            2: GENERATION_MERGE_PARTITION: A partition operation is merged into the respective
            related operation
            and generated as 1 (ie prodedure in HANA).
        generation_group_type : int
            11: GENERATION_GROUP_NONE # No grouping is applied. This means that solution specific
            implementation will define how to deal with this
            12: GENERATION_GROUP_FUNCTIONAL # Grouping is based on functional grouping. Meaning
            that logical related elements such as partiion / fit / and related score will be
            put together
        sda_grant_service:  str
            When generating sda artifacts which grant service can be used to access the right
            grants.
        remote_source : str
            When generating sda artifacts what is the name of the remote source to be used.
        """
        # Remove improper characters
        project_name = StringUtils.remove_special_characters(project_name)
        module_name = project_name
        app_id = module_name
        schema = '"' + (module_name + '_SCHEMA').upper() + '"'
        # This is the root folder in the outputdir where the artifacts will be generated.
        output_path = os.path.join(outputdir, project_name)
        config = ConfigHandler()
        # Populate config
        config.add_entry(ConfigConstants.CONFIG_KEY_OUTPUT_PATH, output_path)
        config.add_entry(ConfigConstants.CONFIG_KEY_PROJECT_NAME, project_name)
        config.add_entry(ConfigConstants.CONFIG_KEY_VERSION, version)
        config.add_entry(ConfigConstants.CONFIG_KEY_GRANT_SERVICE, grant_service)
        config.add_entry(ConfigConstants.CONFIG_KEY_SDA_GRANT_SERVICE, sda_grant_service)
        config.add_entry(ConfigConstants.CONFIG_KEY_SDA_REMOTE_SOURCE, remote_source)
        config.add_entry(ConfigConstants.CONFIG_KEY_MERGE_STRATEGY, generation_merge_type)
        config.add_entry(ConfigConstants.CONFIG_KEY_GROUP_STRATEGY, generation_group_type)
        config.add_entry(ConfigConstants.CONFIG_KEY_OUTPUT_DIR, outputdir)

        config.add_entry(ConfigConstants.CONFIG_KEY_MODULE_NAME, module_name)
        config.add_entry(ConfigConstants.CONFIG_KEY_CDS_CONTEXT, 'output')
        config.add_entry(ConfigConstants.CONFIG_KEY_APPID, app_id)

        config.add_entry(ConfigConstants.CONFIG_KEY_SCHEMA, schema)
        config.add_entry(ConfigConstants.CONFIG_KEY_SQL_PROCESSED, {})

        config.add_entry(ConfigConstants.CONFIG_KEY_DATA_SOURCE_MAPPING, {})
        return config
