CLASS <<AMDP_NAME>> DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    INTERFACES if_hemi_model_management.
    INTERFACES if_hemi_procedure.
    INTERFACES if_amdp_marker_hdb.

    TYPES:
      BEGIN OF ty_train_input,
<<TRAIN_INPUT_STRUCTURE>>
      END OF ty_train_input,
      tt_training_data TYPE STANDARD TABLE OF ty_train_input WITH DEFAULT KEY,
      tt_predict_data  TYPE STANDARD TABLE OF ty_train_input WITH DEFAULT KEY,
      <<CAST_TARGET_OUTPUT>>
      BEGIN OF ty_predict_result,
<<RESULT_OUTPUT_STRUCTURE>>
<<REASON_CODE_STRUCTURE>>
      END OF ty_predict_result,
      tt_predict_result TYPE STANDARD TABLE OF ty_predict_result WITH DEFAULT KEY.

    TYPES:
      BEGIN OF ty_metrics,
        key   TYPE string,
        value TYPE string,
      END OF ty_metrics,
      tt_metrics TYPE STANDARD TABLE OF ty_metrics WITH DEFAULT KEY,
      BEGIN OF ty_model,
        row_index     TYPE int4,
        part_index    TYPE int4,
        model_content TYPE string,
      END OF ty_model,
      tt_model TYPE STANDARD TABLE OF ty_model WITH DEFAULT KEY.

    CLASS-METHODS training
      AMDP OPTIONS READ-ONLY
      IMPORTING
        VALUE(it_data)                TYPE tt_training_data
        VALUE(it_param)               TYPE if_hemi_model_management=>tt_pal_param
      EXPORTING
        VALUE(et_model)               TYPE tt_model
        VALUE(et_confusion_matrix)    TYPE shemi_confusion_matrix_t
        VALUE(et_variable_importance) TYPE shemi_variable_importance_t
        VALUE(et_metrics)             TYPE tt_metrics
        VALUE(et_gen_info)            TYPE tt_metrics
      RAISING
        cx_amdp_execution_failed.

    CLASS-METHODS predict_with_model_version
      AMDP OPTIONS READ-ONLY
      IMPORTING
        VALUE(it_data)   TYPE tt_predict_data
        VALUE(it_model)  TYPE tt_model
        VALUE(it_param)  TYPE if_hemi_model_management=>tt_pal_param
      EXPORTING
        VALUE(et_result) TYPE tt_predict_result
      RAISING
        cx_amdp_execution_failed.

  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS <<AMDP_NAME>> IMPLEMENTATION.

  METHOD if_hemi_model_management~get_amdp_class_name.
    DATA lr_self TYPE REF TO <<AMDP_NAME>>.
    TRY.
        CREATE OBJECT lr_self.
        ev_name = cl_abap_classdescr=>get_class_name( lr_self ).
      CATCH cx_badi_context_error.
      CATCH cx_badi_not_implemented.
    ENDTRY.
  ENDMETHOD.

  METHOD if_hemi_procedure~get_procedure_parameters.
    et_training = VALUE #(
       ( name = 'IT_DATA'                role = cl_hemi_constants=>cs_proc_role-data                         )
       ( name = 'IT_PARAM'               role = cl_hemi_constants=>cs_proc_role-param                        )
       ( name = 'ET_MODEL'               role = cl_hemi_constants=>cs_proc_role-model                        )
       ( name = 'ET_VARIABLE_IMPORTANCE' role = cl_hemi_constants=>cs_proc_role-stats add_info = 'imp'       )
       ( name = 'ET_CONFUSION_MATRIX'    role = cl_hemi_constants=>cs_proc_role-stats add_info = 'confusion' )
       ( name = 'ET_METRICS'             role = cl_hemi_constants=>cs_proc_role-stats add_info = 'metrics'   )
       ( name = 'ET_GEN_INFO'            role = cl_hemi_constants=>cs_proc_role-stats add_info = 'gen'       )
    ).
    et_apply = VALUE #(
       ( name = 'IT_DATA'   role = cl_hemi_constants=>cs_proc_role-data                        )
       ( name = 'IT_MODEL'  role = cl_hemi_constants=>cs_proc_role-model add_info = 'et_model' )
       ( name = 'IT_PARAM'  role = cl_hemi_constants=>cs_proc_role-param                       )
       ( name = 'ET_RESULT' role = cl_hemi_constants=>cs_proc_role-result                      )
    ).
  ENDMETHOD.

  METHOD if_hemi_model_management~get_meta_data.
    es_meta_data-model_parameters = <<PARAMETER>>.
    es_meta_data-model_parameter_defaults = <<PARAMETER_DEFAULT>>.

    es_meta_data-training_data_set = '<<TRAINING_DATASET>>'.
    es_meta_data-apply_data_set    = '<<APPLY_DATASET>>'.

    es_meta_data-field_descriptions = VALUE #( ( name = '<<TARGET_COLUMN>>' role = cl_hemi_constants=>cs_field_role-target )<<KEY_FIELD_DESCRIPTION>> ).
  ENDMETHOD.

  METHOD training BY DATABASE PROCEDURE FOR HDB LANGUAGE SQLSCRIPT OPTIONS READ-ONLY.
    declare complete_statistics table (stat_name nvarchar(256), stat_value nvarchar(1000), class_name nvarchar(256));
    declare complete_metrics    table (metric_name nvarchar(256), x double, y double);
    declare validation_ki double;
    declare train_ki      double;

    /* Step 1. Input data preprocessing (missing values, rescaling, encoding, etc)
               Based on the scenario, add ids, select fields relevant for the training, cast to the appropriate data type, convert nulls into meaningful values.
               Note: decimal must be converted into double.*/

    call _sys_afl.pal_missing_value_handling(:it_data, :it_param, it_data, _);

    /* Step 2. Sampling for training and model quality debriefing;
               Imbalanced up/downsampling + Train/Test/Validation random or stratified partition sampling */

    call _sys_afl.pal_partition(:it_data, :it_param, part_id);

    lt_train = select lt.* from :it_data as lt, :part_id as part_id where lt.id = part_id.id and part_id.partition_type = 1;
    lt_test  = select lt.* from :it_data as lt, :part_id as part_id where lt.id = part_id.id and part_id.partition_type = 2;
    lt_val   = select lt.* from :it_data as lt, :part_id as part_id where lt.id = part_id.id and part_id.partition_type = 3;

    /* Step 3. Unified classification training */

    call _sys_afl.pal_unified_classification(:lt_train, :it_param, :et_model, :et_variable_importance, lt_stat, lt_opt, lt_cm, lt_metrics, lt_ph1, lt_ph2);

    /* Step 4. Unified classification scoring + debriefing additional metrics and gain charts */

    call _sys_afl.pal_unified_classification_score(:lt_train, :et_model, :it_param, result_train, stats_train, cm_train,  metrics_train);
    call _sys_afl.pal_unified_classification_score(:lt_test,  :et_model, :it_param, result_test,  stats_test,  cm_test,   metrics_test);
    call _sys_afl.pal_unified_classification_score(:lt_val,   :et_model, :it_param, result_val,   stats_val,   cm_val,    metrics_val);

    -- output confusion matrix is derived from the validation dataset
    et_confusion_matrix = select * from :cm_val;

    complete_statistics = select concat('VALIDATION_', stat_name) as stat_name, stat_value, class_name from :stats_val
                union all select concat('TEST_',       stat_name) as stat_name, stat_value, class_name from :stats_test
                union all select concat('TRAIN_',      stat_name) as stat_name, stat_value, class_name from :stats_train;

    complete_metrics = select concat('VALIDATION_', "NAME") as metric_name, x, y from :metrics_val
             union all select concat('TEST_',       "NAME") as metric_name, x, y from :metrics_test
             union all select concat('TRAIN_',      "NAME") as metric_name, x, y from :metrics_train;

    -- Calculate KI and KR and other key metrics
    select to_double(stat_value) * 2 - 1 into validation_ki from :complete_statistics where stat_name = 'VALIDATION_AUC';
    select to_double(stat_value) * 2 - 1 into train_ki      from :complete_statistics where stat_name = 'TRAIN_AUC';

    et_metrics = select 'PredictivePower'      as key, to_nvarchar(:validation_ki)                        as value from dummy
       union all select 'PredictionConfidence' as key, to_nvarchar(1.0 - abs(:validation_ki - :train_ki)) as value from dummy
    -- Provide metrics that are displayed in the quality information section of a model version in the ISLM Intelligent Scenario Management app
    /* <<<<<< TODO: Starting point of adaptation */
       union all select 'AUC'                  as key, stat_value                                         as value from :complete_statistics
                    where stat_name = 'VALIDATION_AUC';
    /* <<<<<< TODO: End point of adaptation */

    gain_chart = select gerneral.X,
                        (select min(y) from :complete_metrics as train_col      where metric_name = 'TRAIN_CUMGAINS'      and gerneral.x = train_col.x     ) as train,
                        (select min(y) from :complete_metrics as validation_col where metric_name = 'VALIDATION_CUMGAINS' and gerneral.x = validation_col.x) as validation,
                        (select min(y) from :complete_metrics as test_col       where metric_name = 'TEST_CUMGAINS'       and gerneral.x = test_col.x      ) as test,
                        (select min(y) from :complete_metrics as wizard_col     where metric_name = 'TRAIN_PERF_CUMGAINS' and gerneral.x = wizard_col.x    ) as wizard
                 from :complete_metrics as gerneral where gerneral.metric_name like_regexpr '(TRAIN|VALIDATION|TEST)_CUMGAINS' group by gerneral.x order by gerneral.x asc;

    gain_chart = select t1.x, t1.train, t1.validation, t1.test, coalesce(t1.wizard, t2.wizard) as wizard from :gain_chart as t1
        left outer join :gain_chart as t2 on t2.x = ( select max(t3.x) from :gain_chart as t3 where t3.x < t1.x and t3.wizard is not null);
    gain_chart = select t1.x, coalesce(t1.train, t2.train) as train, t1.validation, t1.test, t1.wizard from :gain_chart as t1
        left outer join :gain_chart as t2 on t2.x = ( select max(t3.x) from :gain_chart as t3 where t3.x < t1.x and t3.train is not null);
    gain_chart = select t1.x, t1.train, coalesce(t1.validation, t2.validation) as validation, t1.test, t1.wizard from :gain_chart as t1
        left outer join :gain_chart as t2 on t2.x = ( select max(t3.x) from :gain_chart as t3 where t3.x < t1.x and t3.validation is not null);
    gain_chart = select t1.x, t1.train, t1.validation, coalesce(t1.test, t2.test) as test, t1.wizard from :gain_chart as t1
        left outer join :gain_chart as t2 on t2.x = ( select max(t3.x) from :gain_chart as t3 where t3.x < t1.x and t3.TEST is not null) order by x;

    et_gen_info = select 'HEMI_Profitcurve' as key,
                         '{ "Type": "detected", "Frequency" : "' || x || '", "Random" : "' || x || '", "Wizard": "' || wizard || '", "Estimation": "' || train || '", "Validation": "' || validation || '", "Test": "' || test || '"}' as value
                         from :gain_chart
    -- Provide metrics that are displayed in the general additional info section of a model version in the ISLM Intelligent Scenario Management app
    /* <<<<<< TODO: Starting point of adaptation */
        union all select stat_name as key, stat_value as value from :complete_statistics where class_name is null;
    /* <<<<<< TODO: End point of adaptation */
  ENDMETHOD.

  METHOD predict_with_model_version BY DATABASE PROCEDURE FOR HDB LANGUAGE SQLSCRIPT OPTIONS READ-ONLY.

    /* Step 1. Input data preprocessing (missing values, rescaling, encoding, etc).
               Note: the input data preprocessing must correspond with the one in the training method.
               Based on the scenario, add ids, select fields relevant for the training, cast to the appropriate data type, convert nulls into meaningful values.
               Note: decimal must be converted into double. */
    lt_data = select <<PREDICT_DATA_COLS>> from :it_data;

    call _sys_afl.pal_missing_value_handling(:lt_data, :it_param, lt_data_predict, lt_placeholder1);

    /* Step 2. Execute prediction */

    call _sys_afl.pal_unified_classification_predict(:lt_data_predict, :it_model, :it_param, lt_result, lt_placeholder2);

    /* Step 3. Map prediction results back to the composite key */

    et_result = select <<RESULT_FIELDS>>,
<<RESULT_REASON_CODE_FIELDS>>
                from :lt_result as result;
  ENDMETHOD.

ENDCLASS.
