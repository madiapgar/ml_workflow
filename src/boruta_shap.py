## 3
## optional: running boruta shap for important feature selection on the dataset

## needed libraries
import argparse
import pandas as pd
import numpy as np
from BorutaShap import BorutaShap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)

## functions
## argparse function
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metadata",
                        help="Path to the machine-learning approved metadata file as a .tsv.",
                        default="None",
                        type=str)
    parser.add_argument("-o", "--otu_table",
                        help="Path to a .tsv file that contains the data to train the models.",
                        default="None",
                        type=str)
    parser.add_argument("-s", "--sample_id",
                        help="The name of the sample id column in the metadata and otu table.",
                        default="None",
                        type=str)
    parser.add_argument("-p", "--predict_col",
                        help="The name of the column you want to predict with your models in the metadata.",
                        default="None",
                        type=str)
    parser.add_argument("-r", "--train_feat_out",
                        help="Path to where the accepted features from the training data will be saved as a .tsv file.",
                        default="None",
                        type=str) 
    parser.add_argument("-t", "--test_feat_out",
                        help="Path to where the accepted features from the testing data will be saved as a .tsv file.",
                        default="None",
                        type=str)
    return parser.parse_args()


## generate x and y dataframes for machine learning and/or boruta shap/kfold cross validation
def make_xy_tables(meta_df,
                   otu_df,
                   merge_on,
                   y_col):
    output_dict = {}
    
    mini_meta = meta_df.loc[:, (merge_on, y_col)]
    comb_df = otu_df.merge(mini_meta, how="left", on=[merge_on])
    ## x - the side that has the data I want the model to use to predict y
    pre_x_df = comb_df.copy()
    x_df = pre_x_df.drop(y_col, axis=1)
    x_df[merge_on] = x_df[merge_on].astype(float)
    x_df = x_df.drop(merge_on, axis=1)
    ## y - what is to be predicted
    y_df = comb_df[y_col]

    ## saving my outputs
    output_dict.update({"x_dataframe": x_df,
                        "y_dataframe": y_df})
    return(output_dict)


## run kfold cross validation on x and y datasets and then identify important features via boruta shap
def kfold_boruta_shap(k_fold,
                      feature_selector,
                      x_dataframe,
                      y_dataframe,
                      trial_num):
    output_dict = {}

    train_list = []
    test_list = []
    for i, (train_index, test_index) in enumerate(k_fold.split(x_dataframe, y_dataframe)):
        print(f"Fold {i}:")
        print(f"Training dataset index: {train_index}")
        print(f"Testing dataset index: {test_index}")
        ## setting up test/train datasets 
        x_train = x_dataframe.filter(items=train_index, axis=0)
        x_test = x_dataframe.filter(items=test_index, axis=0)
        y_train = y_dataframe.filter(items=train_index, axis=0)
        y_test = y_dataframe.filter(items=test_index, axis=0)

        ## running boruta shap on training data
        feature_selector.fit(X=x_train,
                             y=y_train, 
                             n_trials=trial_num,
                             random_state=0,
                             sample=False,
                             verbose=True)
        
        train_acc_features = feature_selector.accepted
        train_list = train_list + train_acc_features
        
        ## running borta shap on testing data
        feature_selector.fit(X=x_test,
                             y=y_test, 
                             n_trials=trial_num,
                             random_state=0,
                             sample=False,
                             verbose=True)
        
        test_acc_features = feature_selector.accepted
        test_list = test_list + test_acc_features

    ## saving my outputs
    output_dict.update({"acc_train": train_list,
                        "acc_test": test_list})
    return(output_dict)


## to make the output table of the kfold cross validated boruta shap results
def create_occurence_table(input_list):
    wanted_df = pd.DataFrame(np.unique(input_list, return_counts=True)).T
    wanted_df.columns = ["feature", "num_occurences"]
    wanted_df = wanted_df.sort_values(by="num_occurences", ascending=False)
    wanted_df["av_occurences"] = wanted_df["num_occurences"]/5
    return(wanted_df)



## cross validator and boruta shap 
kf = KFold(n_splits=5)
grad_boost = GradientBoostingClassifier(n_estimators=100)
random_forest_bs = BorutaShap(importance_measure='shap', 
                              classification=False)
grad_boost_bs = BorutaShap(model=grad_boost,
                           importance_measure='shap',
                           classification=True)
borutaShap_dict = {"random_forest": random_forest_bs}



## reading in files and data wrangling
args = get_args()
ml_data_df = pd.read_csv(args.otu_table, sep='\t')
meta = pd.read_csv(args.metadata, sep='\t')

## adding gradient boosted classifier to boruta shap if the y pred column is binary
if len(meta[args.predict_col].unique()) == 2:
    borutaShap_dict.update({"grad_boost": grad_boost_bs})

## x - the side that has the data I want the model to use to predict y
## y - what is to be predicted
## making sure that the metadata can be matched back up to the y preds regardless of order
meta_ordered = meta.sort_values(by=args.sample_id, ascending=True)
meta_ordered = meta_ordered.drop("Unnamed: 0", axis=1)
ordered_ml_df = ml_data_df.sort_values(by=args.sample_id, ascending=True)

if 'Unnamed: 0' in ordered_ml_df.columns:
    ordered_ml_df = ordered_ml_df.drop("Unnamed: 0", axis=1)
else:
    pass

xy_results = make_xy_tables(meta_df=meta_ordered,
                            otu_df=ordered_ml_df,
                            merge_on=args.sample_id,
                            y_col=args.predict_col)

x_dataframe = xy_results["x_dataframe"]
y_dataframe = xy_results["y_dataframe"]



## for loop to run boruta shap on the data !!
bs_acc_train = {}
bs_acc_test = {}
for label, boruta_shap in borutaShap_dict.items():
    bs_results = kfold_boruta_shap(k_fold=kf,
                                   feature_selector=boruta_shap,
                                   x_dataframe=x_dataframe,
                                   y_dataframe=y_dataframe,
                                   trial_num=100)
    
    ## pulling out boruta shap accepted features 
    ## training data
    bs_train_list = bs_results["acc_train"]
    bs_train_df = create_occurence_table(input_list=bs_train_list)
    bs_train_df["bs_model"] = label

    ## testing data
    bs_test_list = bs_results["acc_test"]
    bs_test_df = create_occurence_table(input_list=bs_test_list)
    bs_test_df["bs_model"] = label

    bs_acc_train.update({f"{label}_accepted": bs_train_df})
    bs_acc_test.update({f"{label}_accepted": bs_test_df})


train_features = pd.concat(bs_acc_train, ignore_index=True)
test_features = pd.concat(bs_acc_test, ignore_index=True)



## saving my outputs
train_features.to_csv(args.train_feat_out, sep="\t")
test_features.to_csv(args.test_feat_out, sep="\t")




