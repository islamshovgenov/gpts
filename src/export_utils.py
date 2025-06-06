from docx_tools import (
    export_individual_pk_tables,
    export_auc_residual_tables,
    export_log_transformed_pk_tables,
    export_sas_anova_report,
    export_log_ci_tables,
    export_be_result_table,
    export_power_analysis_table,
)


def export_be_tables(pk_table, pivot, test_name, ref_name, substance, dose_test, dose_ref):
    paths = {}
    paths["individual_pk"] = export_individual_pk_tables(
        pk_table,
        test_name=test_name,
        ref_name=ref_name,
        substance=substance,
        dose_test=dose_test,
        dose_ref=dose_ref,
        save_path="Индивидуальные_PK_таблицы.docx",
    )
    paths["auc_residual"] = export_auc_residual_tables(
        df=pk_table,
        test_name=test_name,
        ref_name=ref_name,
        substance=substance,
        dose_test=dose_test,
        dose_ref=dose_ref,
        save_path="AUC_residual_tables.docx",
    )
    paths["ln_pk"] = export_log_transformed_pk_tables(
        pk_df=pk_table,
        test_name=test_name,
        ref_name=ref_name,
        substance=substance,
        dose_test=dose_test,
        dose_ref=dose_ref,
        save_path="ln_PK_values.docx",
    )
    paths["anova_report"] = export_sas_anova_report(
        pk_df=pk_table,
        substance=substance,
        dose_test=dose_test,
        save_path="ANOVA_SAS_Report.docx",
    )
    paths["log_ci"] = export_log_ci_tables(
        df=pivot,
        substance=substance,
        save_path="log_CI_tables.docx",
    )
    return paths


def export_be_result(gmr_list, ci_low_list, ci_up_list, cv_list):
    return export_be_result_table(
        gmr_list=gmr_list,
        ci_low_list=ci_low_list,
        ci_up_list=ci_up_list,
        cv_list=cv_list,
    )


def export_power_table(gmr_list, cv_list, n):
    return export_power_analysis_table(
        gmr_list=gmr_list,
        cv_list=cv_list,
        n=n,
    )
