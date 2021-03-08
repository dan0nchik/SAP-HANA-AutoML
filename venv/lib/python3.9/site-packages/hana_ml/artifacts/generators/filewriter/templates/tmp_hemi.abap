CLASS <<AMDP_NAME>> DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    INTERFACES if_hemi_model_management.
    INTERFACES if_hemi_procedure.
    INTERFACES if_amdp_marker_hdb.
    TYPES:
    begin of st_train_input,
    <<TRAIN_INPUT_STRUCTURE>>
    end of st_train_input,
    tt_training_data TYPE STANDARD TABLE OF st_train_input WITH DEFAULT KEY,
    tt_predict_data  TYPE STANDARD TABLE OF st_train_input WITH DEFAULT KEY,
    <<CAST_TARGET_OUTPUT>>
    begin of st_result,
    <<RESULT_OUTPUT_STRUCTURE>>
    end of st_result,
    tt_result TYPE STANDARD TABLE OF st_result WITH DEFAULT KEY.
	TYPES:
	<<TRAIN_OUTPUT_TYPE_DEFINITIONS>>
	.
    CLASS-METHODS training
        AMDP OPTIONS READ-ONLY
    IMPORTING
      VALUE(it_data)                TYPE tt_training_data
      VALUE(it_param)               TYPE if_hemi_model_management=>tt_pal_param
    EXPORTING
      <<TRAIN_METHOD_SIGNATURE>>
    RAISING
      cx_amdp_execution_failed .

    CLASS-METHODS predict_with_model_version
        AMDP OPTIONS READ-ONLY
    IMPORTING
      VALUE(it_data)   TYPE tt_predict_data
      VALUE(it_model)  TYPE <<MODEL_STRUCTURE>>
      VALUE(it_param)  TYPE if_hemi_model_management=>tt_pal_param
    EXPORTING
      VALUE(et_result) TYPE tt_result
    RAISING
      cx_amdp_execution_failed .

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

  METHOD if_hemi_model_management~get_meta_data.
    es_meta_data-model_parameters = <<PARAMETER>>.
    es_meta_data-model_parameter_defaults = <<PARAMETER_DEFAULT>>.
    es_meta_data-field_descriptions = VALUE #( ( name = '<<TARGET_COLUMN>>' role = cl_hemi_constants=>cs_field_role-target )<<KEY_FIELD_DESCRIPTION>> ).
  ENDMETHOD.

  METHOD PREDICT_WITH_MODEL_VERSION BY DATABASE PROCEDURE FOR HDB LANGUAGE SQLSCRIPT OPTIONS SUPPRESS SYNTAX ERRORS READ-ONLY.
    DECLARE max_occurences INT;
    SELECT MAX( OCCURRENCES_REGEXPR(',' IN stringargs ) + 1 ) INTO max_occurences FROM :it_param;
    it_param = SELECT DISTINCT TO_NVARCHAR(name) AS NAME, TO_INT(intargs) AS INTARGS, TO_DOUBLE(doubleargs) AS DOUBLEARGS,
    TO_NVARCHAR(SUBSTR_REGEXPR('[^, ]+' IN stringargs OCCURRENCE SERIES."ELEMENT_NUMBER")) AS stringargs
             FROM :it_param CROSS JOIN public.SERIES_GENERATE_INTEGER(1,0, :max_occurences) SERIES
             WHERE NOT (SUBSTR_REGEXPR('[^, ]+' IN stringargs OCCURRENCE SERIES."ELEMENT_NUMBER") IS NULL AND doubleargs is NULL AND intargs is NULL);
    CALL <<PAL_PREDICT_CALL>>(:it_data, :it_model, :it_param, lt_result<<ADDITIONAL_PREDICT_OUTPUTS>>);
    et_result = SELECT <<RESULT_FIELDS>> from :lt_result as result;
  ENDMETHOD.

  METHOD TRAINING BY DATABASE PROCEDURE FOR HDB LANGUAGE SQLSCRIPT OPTIONS SUPPRESS SYNTAX ERRORS READ-ONLY.
    DECLARE max_occurences INT;
    SELECT MAX( OCCURRENCES_REGEXPR(',' IN stringargs ) + 1 ) INTO max_occurences FROM :it_param;
    it_param = SELECT DISTINCT TO_NVARCHAR(name) AS NAME, TO_INT(intargs) AS INTARGS, TO_DOUBLE(doubleargs) AS DOUBLEARGS,
    TO_NVARCHAR(SUBSTR_REGEXPR('[^, ]+' IN stringargs OCCURRENCE SERIES."ELEMENT_NUMBER")) AS stringargs
             FROM :it_param CROSS JOIN public.SERIES_GENERATE_INTEGER(1,0, :max_occurences) SERIES
             WHERE NOT (SUBSTR_REGEXPR('[^, ]+' IN stringargs OCCURRENCE SERIES."ELEMENT_NUMBER") IS NULL AND doubleargs is NULL AND intargs is NULL);
    CALL <<PAL_TRAIN_CALL>>(:it_data, :it_param, <<SQL_OUTPUT_VARIABLES>>);
  ENDMETHOD.
  
  METHOD if_hemi_procedure~get_procedure_parameters.
    et_training = VALUE #(
       ( name = 'IT_DATA'                role = cl_hemi_constants=>cs_proc_role-data  )
       ( name = 'IT_PARAM'               role = cl_hemi_constants=>cs_proc_role-param )
		<<PROCEDURE_PARAMETERS>>
      ).
    et_apply = VALUE #(
       ( name = 'IT_DATA'   role = cl_hemi_constants=>cs_proc_role-data   )
       ( name = 'IT_MODEL'  role = cl_hemi_constants=>cs_proc_role-model add_info = 'et_model' )
       ( name = 'IT_PARAM'  role = cl_hemi_constants=>cs_proc_role-param  )
       ( name = 'ET_RESULT' role = cl_hemi_constants=>cs_proc_role-result )
      ).
  ENDMETHOD.
ENDCLASS.