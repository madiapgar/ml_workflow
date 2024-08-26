import os

## global variables from the config file
## overarching variables 
PROJECT = config["project_name"]
BORUTA_STAT = config["boruta_shap"]
CONDA_ENV = config["conda_env"]

## variables for inputs 
META = config["metadata"]
OTU = config["otu_table"]
META_DICT = config["meta_dictionary"]
SAMPLE_ID = config["sample_id_col"]
PREDICT_COL = config["y_predict_col"]

## pipeline output list assembly
def comb_filepaths(filepath1,
                   filepath2):
    return os.path.join(filepath1, filepath2)

out_list = ["best_performing_models.tsv",
            "model_predict_scores.tsv",
            "comb_meta_yPred.tsv"]

rule_all_list = [comb_filepaths(PROJECT, filepath) for filepath in out_list]


if BORUTA_STAT == True:
    ## adding boruta shap rule outputs to the rule all list if it's true
    boruta_out_list = ["borutaShap_train_feat.tsv",
                        "borutaShap_test_feat.tsv"]
    boruta_out_list = [comb_filepaths(PROJECT, filepath) for filepath in boruta_out_list]
    rule_all_list.append(boruta_out_list)


rule all:
    input:
        rule_all_list


rule find_best_models:
    input:
        metadata = META,
        otu_table = OTU,
        meta_dict = META_DICT
    output:
        top_models_out = os.path.join(PROJECT, "best_performing_models.tsv")
    conda:
        CONDA_ENV
    params:
        sample_id = SAMPLE_ID,
        predict_col = PREDICT_COL
    shell:
        """
        python "src/best_models.py" --metadata {input.metadata} \
                                    --otu_table {input.otu_table} \
                                    --metadata_dict {input.meta_dict} \
                                    --sample_id {params.sample_id} \
                                    --predict_col {params.predict_col} \
                                    --best_models_out {output.top_models_out}
        """

rule predict_y_values:
    input:
        metadata = META,
        otu_table = OTU,
        meta_dict = META_DICT,
        best_models_in = os.path.join(PROJECT, "best_performing_models.tsv")
    output:
        model_scores = os.path.join(PROJECT, "model_predict_scores.tsv"),
        comb_yPreds = os.path.join(PROJECT, "comb_meta_yPred.tsv")
    conda:
        CONDA_ENV
    params:
        sample_id = SAMPLE_ID,
        predict_col = PREDICT_COL
    shell:
        """
        python "src/model_predict.py" --metadata {input.metadata} \
                                      --otu_table {input.otu_table} \
                                      --metadata_dict {input.meta_dict} \
                                      --sample_id {params.sample_id} \
                                      --predict_col {params.predict_col} \
                                      --best_models_in {input.best_models_in} \
                                      --model_res_out {output.model_scores} \
                                      --comb_yPreds_out {output.comb_yPreds}
        """


rule run_boruta_shap:
    input:
        metadata = META,
        otu_table = OTU
    output:
        train_feat_out = os.path.join(PROJECT, "borutaShap_train_feat.tsv"),
        test_feat_out = os.path.join(PROJECT, "borutaShap_test_feat.tsv")
    conda:
        CONDA_ENV
    params:
        sample_id = SAMPLE_ID,
        predict_col = PREDICT_COL 
    shell:
        """
        python "src/boruta_shap.py" --metadata {input.metadata} \
                                    --otu_table {input.otu_table} \
                                    --sample_id {params.sample_id} \
                                    --predict_col {params.predict_col} \
                                    --train_feat_out {output.train_feat_out} \
                                    --test_feat_out {output.test_feat_out}
        """