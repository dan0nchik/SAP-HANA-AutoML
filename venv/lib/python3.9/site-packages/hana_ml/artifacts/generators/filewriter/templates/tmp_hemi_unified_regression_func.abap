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
        VALUE(et_metrics)             TYPE tt_metrics
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
       ( name = 'ET_METRICS'             role = cl_hemi_constants=>cs_proc_role-stats add_info = 'metrics'   )
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

    /* Step 3. Unified regression training */

    call _sys_afl.pal_unified_regression(:lt_train, :it_param, :et_model, lt_stat, lt_opt, lt_partition, lt_ph1, lt_ph2);

    /* Step 4. Unified regression scoring + debriefing additional metrics and gain charts */

    call _sys_afl.pal_unified_regression_score(:lt_train, :et_model, :it_param, result_train, stats_train);
    call _sys_afl.pal_unified_regression_score(:lt_test,  :et_model, :it_param, result_test,  stats_test);
    call _sys_afl.pal_unified_regression_score(:lt_val,   :et_model, :it_param, result_val,   stats_val);

    complete_statistics = select concat('VALIDATION_', stat_name) as stat_name, stat_value, class_name from :stats_val
                union all select concat('TEST_',       stat_name) as stat_name, stat_value, class_name from :stats_test
                union all select concat('TRAIN_',      stat_name) as stat_name, stat_value, class_name from :stats_train;

  ENDMETHOD.

  METHOD predict_with_model_version BY DATABASE PROCEDURE FOR HDB LANGUAGE SQLSCRIPT OPTIONS READ-ONLY.

    /* Step 1. Input data preprocessing (missing values, rescaling, encoding, etc).
               Note: the input data preprocessing must correspond with the one in the training method.
               Based on the scenario, add ids, select fields relevant for the training, cast to the appropriate data type, convert nulls into meaningful values.
               Note: decimal must be converted into double. */
    lt_data = select <<PREDICT_DATA_COLS>> from :it_data;

    call _sys_afl.pal_missing_value_handling(:lt_data, :it_param, lt_data_predict, lt_placeholder1);

    /* Step 2. Execute prediction */

    call _sys_afl.pal_unified_regression_predict(:lt_data_predict, :it_model, :it_param, lt_result, lt_placeholder2);

    /* Step 3. Map prediction results back to the composite key */

    et_result = select <<RESULT_FIELDS>>
                from :lt_result as result;
  ENDMETHOD.

ENDCLASS.
