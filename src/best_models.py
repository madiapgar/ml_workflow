## 1
## figuring out the best models to use with the dataset

## needed libraries
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import Counter

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
    parser.add_argument("-d", "--metadata_dict",
                        help="Path to a .tsv file that contains the dictionary keys for the metadata.",
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
    parser.add_argument("-b", "--best_models_out",
                        help="Path to where the best models dataframe will be saved as a .tsv file.",
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

## will run the desired ml model (or list of models via a for loop)
## output is a python dictionary (aka "named list") of a few dataframes
def run_models(wanted_model,
               x_train,
               y_train,
               x_test):
    
    model_out = {}
    wanted_model.fit(x_train, y_train)
    model_y_pred = wanted_model.predict(x_test)
    acc_model = round(wanted_model.score(x_train, y_train) * 100, 2)
    model_out.update({"y_pred": model_y_pred,
                      "acc_score": acc_model})
    return(model_out)

## will take the results from the run_models function above and nicely wrap them up in a couple
## dataframes for model performance comparison
## output is also a python dictionary ("named list" for all intents and purposes)
def model_results(model_name_list,
                  model_scores,
                  model_y_preds,
                  y_test,
                  value_dict):
    output_list = {}
    ## putting accuracy scores into a df
    model_score_df = pd.DataFrame({
        "model": model_name_list,
        "score": model_scores
    })

    ## putting model y preds in a df with the key 
    y_pred_df = pd.DataFrame(model_y_preds).T
    y_pred_df.columns = model_name_list
    y_pred_df["key"] = y_test.values
    y_pred_df = y_pred_df.set_index(y_test.index)
    
    ## counting how many of each value were predicted and mapping
    ## them to their categorical counterpart
    count_y_pred = []
    for columns in y_pred_df:
        y_pred_df[columns] = y_pred_df[columns].map(value_dict)
        count_y_pred.append(Counter(y_pred_df[columns]))
    
    ## putting accuracy scores and number of y preds in the same table 
    update_model_names = model_name_list.copy()
    update_model_names.append("key")
    count_y_pred_df = pd.DataFrame(count_y_pred)
    count_y_pred_df["model"] = update_model_names
    count_scores_df = model_score_df.merge(count_y_pred_df, how="right", on=["model"])

    ## saving my function outputs in a list
    output_list.update({"y_preds": y_pred_df,
                        "acc_count_table": count_scores_df})
    return(output_list)



## saving models as a variable and putting them in a list
logreg = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier(n_neighbors=3)
gaussian = GaussianNB()
perceptron = Perceptron()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)
grad_boost = GradientBoostingClassifier(n_estimators=100)
ridge_class = RidgeClassifierCV()

model_list = [logreg, svc, knn, gaussian, perceptron, decision_tree,
              random_forest, grad_boost, ridge_class]
model_labs = ["logreg", "svc", "knn", "gaussian", "perceptron", "decision_tree",
              "random_forest", "grad_boost", "ridge_class"]
model_dict = dict(zip(model_labs, model_list))
## 5 splits gives 80/20 distribution
kf = KFold(n_splits=5)



## reading in files and data wrangling
args = get_args()
ml_data_df = pd.read_csv(args.otu_table, sep='\t')
meta = pd.read_csv(args.metadata, sep='\t')
meta_keys = pd.read_csv(args.metadata_dict, sep='\t')

inverse_dict = dict(zip(meta_keys["assigned_num"], meta_keys[args.predict_col]))



## x - the side that has the data I want the model to use to predict y
## y - what is to be predicted
xy_results = make_xy_tables(meta_df=meta,
                            otu_df=ml_data_df,
                            merge_on=args.sample_id,
                            y_col=args.predict_col)

x_dataframe = xy_results["x_dataframe"]
y_dataframe = xy_results["y_dataframe"]



## for loop to compute best models to use
output = {}

for i, (train_index, test_index) in enumerate(kf.split(x_dataframe, y_dataframe)):
    print(f"Fold {i}:")
    print(f"Training dataset index: {train_index}")
    print(f"Testing dataset index: {test_index}")
    ## setting up test/train datasets 
    x_train = x_dataframe.filter(items=train_index, axis=0)
    x_test = x_dataframe.filter(items=test_index, axis=0)
    y_train = y_dataframe.filter(items=train_index, axis=0)
    y_test = y_dataframe.filter(items=test_index, axis=0)

    ## running my various models on all 5 datasets
    y_pred_output = []
    score_output = []
    model_name_output = []
    for label, model in model_dict.items():
        model_scores = run_models(model,
                                  x_train=x_train,
                                  y_train=y_train,
                                  x_test=x_test)
        y_pred_results = model_scores["y_pred"]
        score_results = model_scores["acc_score"]

        y_pred_output.append(y_pred_results)
        score_output.append(score_results)
        model_name_output.append(label)
    
    ## non shuffled data model results - y preds and accuracy scores by model
    ## so I can decide which one to use moving forward w my actual ML model
    all_model_results = model_results(model_name_list=model_name_output,
                                      model_scores=score_output,
                                      model_y_preds=y_pred_output,
                                      y_test=y_test,
                                      value_dict=inverse_dict)
    yPreds = all_model_results["y_preds"]
    countScore = all_model_results["acc_count_table"].fillna(0)
    countScore["fold"] = f"f{i}"

    output.update({f"fold{i}_results": countScore})



## putting all fold results for the models together in one table
fold_model_results = pd.concat(output, ignore_index=True)

## pulling out models that scored above 95 across all folds 
best_model_results = fold_model_results.loc[fold_model_results['score'] > 95]

## taking the models that scored above 95 and counting how many times that model appears across all folds
## this is all to inform which models to use moving forward
top_models = pd.DataFrame(np.unique(best_model_results["model"], return_counts=True)).T
top_models.columns = ["model", "num_occurences"]
top_models = top_models.sort_values(by="num_occurences", ascending=False)



## saving my outputs 
top_models.to_csv(args.best_models_out, sep="\t")





