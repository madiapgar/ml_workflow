## 2
## taking the best models we found and running them through a voter classifier for the best y predictions

## needed libraries
import argparse
import pandas as pd
import numpy as np
import random as rnd
import time
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
from sklearn.ensemble import VotingClassifier

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
    parser.add_argument("-b", "--best_models_in",
                        help="Path to the best models dataframe .tsv file.",
                        default="None",
                        type=str)
    parser.add_argument("-r", "--model_res_out",
                        help="Path to where the model score dataframe will be saved as a .tsv file.",
                        default="None",
                        type=str) 
    parser.add_argument("-c", "--comb_yPreds_out",
                        help="Path to where the combined model predicted y-values dataframe will be saved as a .tsv file.",
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

## grid search to optimize chosen models
## chooses the best parameters for the model to use for prediction
def model_grid_search(model,
                      param_dict,
                      x_dev,
                      y_dev,
                      model_name,
                      x_eval,
                      y_eval):
    output_list = {}
    dict_clf = {}
    GS = GridSearchCV(model,
                      param_dict,
                      cv=4)

    # Fit the data and record time taking to train
    t0 = time.time()
    GS.fit(x_dev, y_dev)
    t = time.time() - t0

    # Store best parameters, score and estimator
    best_clf = GS.best_estimator_
    best_params = GS.best_params_
    best_score = GS.best_score_
    name = model_name

    best_clf.fit(x_dev, y_dev)
    acc_eval = accuracy_score(y_eval, best_clf.predict(x_eval))
    dict_clf[name] = {
        'best_par': best_params,
        'best_clf': best_clf,
        'best_score': best_score,
        'score_eval': acc_eval,
        'fit_time': t,
    }

    ## saving my outputs
    output_list.update({"acc_eval": acc_eval,
                        "dict_clf": dict_clf})
    return(output_list)

## model voting classifier for final y pred results
def model_voteClass(estimator_list,
                    x_train,
                    y_train,
                    x_test,
                    y_test):
    voter = VotingClassifier(estimators=estimator_list, voting='hard')
    voter.fit(x_train, y_train)
    final_y_pred = voter.predict(x_test).astype(int)
    end_comp = pd.DataFrame({
        "key": y_test,
        "final_y_pred": final_y_pred})
    return(end_comp) 

## puts above functions together to run grid search on kfold cross validated x/y train to get average 
## accuracy score for the given model
def kfold_model_predict(x_dataframe,
                        y_dataframe,
                        k_fold,
                        wanted_model,
                        paramgrid,
                        wanted_model_name,
                        add_estimator_list):
    output = {}

    model_y_pred = {}
    mean_acc = {}
    for i, (train_index, test_index) in enumerate(k_fold.split(x_dataframe, y_dataframe)):
        print(f"Fold {i}:")
        print(f"Training dataset index: {train_index}")
        print(f"Testing dataset index: {test_index}")
        ## setting up test/train datasets 
        x_train = x_dataframe.filter(items=train_index, axis=0)
        x_test = x_dataframe.filter(items=test_index, axis=0)
        y_train = y_dataframe.filter(items=train_index, axis=0)
        y_test = y_dataframe.filter(items=test_index, axis=0)

        ## splitting training set to development and evaluation dfs
        x_dev,x_eval,y_dev,y_eval=train_test_split(x_train,
                                                   y_train,
                                                   test_size=0.2,
                                                   random_state=42)
        
        ## grid search 
        grid_search = model_grid_search(model=wanted_model,
                                        param_dict=paramgrid,
                                        x_dev=x_dev,
                                        y_dev=y_dev,
                                        model_name=wanted_model_name,
                                        x_eval=x_eval,
                                        y_eval=y_eval)
        
        dict_clf = grid_search["dict_clf"]
        pre_estimators = [(wanted_model_name, dict_clf[wanted_model_name]['best_clf'])]
        estimators = pre_estimators + add_estimator_list

        y_pred = model_voteClass(estimator_list=estimators,
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_test=x_test,
                                 y_test=y_test)
    
        ## seeing how accurate the model was at predicting which mice had blooms v not 
        y_pred["model_correct"] = np.where(y_pred["key"] == y_pred["final_y_pred"], 1, 0)
        y_pred["fold"] = f"f{i}"
        model_mean_acc = y_pred["model_correct"].mean()

        ## putting together accuracy scores for each fold with the model
        mean_acc.update({f"f{i}": model_mean_acc})
        model_y_pred.update({f"f{i}_yPred": y_pred})

    ## output list
    output.update({"model_mean_acc": mean_acc,
                   "model_y_pred": model_y_pred})
    return(output)



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



## creating a dictionary of parameters for every single model :(
## random forest
rf_paramgrid = {
    'n_estimators':      [100, 150, 200, 250, 300, 400, 500],
    'criterion':         ['gini', 'entropy'],
    'max_features':      ['auto', 'log2'],
    'min_samples_leaf':  list(range(2, 8)),
    'random_state':      [42]
}
## gradient boosted classifier
gb_paramgrid = {
    'n_estimators':      [100, 150, 200, 250, 300, 400, 500],
    'criterion':         ['friedman_mse', 'squared_error'],
    'max_features':      ['sqrt', 'log2'],
    'min_samples_leaf':  list(range(2, 8)),
    'random_state':      [42]
}
## decision tree
dt_paramgrid = {
    'criterion':         ['gini', 'entropy'],
    'max_features':      ['sqrt', 'log2'],
    'min_samples_leaf':  list(range(2, 8)),
    'random_state':      [42]
}
## support vector machine
## this one is funky idk what the best way to set this up is 
svc_paramgrid = {
    'C':                [0.0001, 0.001, 0.01, 0.1, 1.0],
    'gamma':            [0.01, 0.1],
    'random_state':      [42]
} 
## ridge classifier cv
rc_paramgrid = {
    'alphas':        [0.0001, 0.001, 0.01, 0.1, 1.0],
    'scoring':       ['neg_mean_squared_error', 'neg_mean_squared_log_error']
}
## logistic regression
lr_paramgrid = {
    'penalty':      ['None', 'l2'],
    'C':            [0.0001, 0.001, 0.01, 0.1, 1.0],
    'solver':       ['lbfgs', 'newton-cg', 'sag'],
    'random_state':      [42]
}
## knn
knn_paramgrid = {
    'n_neighbors':  [3, 4, 5],
    'weights':      ['uniform', 'distance'],
    'algorithm':    ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size':    [10, 20, 30, 40, 50],
    'p':            [2]
}
## gaussian nb classifier
## there are like no hyperparameters for this model lol idk what to do here
gnb_paramgrid = {
    'priors':      ['none'],
    'var_smoothing': [1e-09],
    'random_state':      [42]
}
## perceptron
per_paramgrid = {
    'penalty':   ['None', 'l2'],
    'alpha':     [0.0001, 0.001, 0.01, 0.1, 1.0],
    'l1_ratio':  [0.15, 0],
    'random_state':      [42]
}

## putting them all into a dictionary together :)
paramgrid_dict = {"random_forest": rf_paramgrid,
                  "grad_boost": gb_paramgrid,
                  "decision_tree": dt_paramgrid,
                  "svc": svc_paramgrid,
                  "ridge_class": rc_paramgrid,
                  "logreg": lr_paramgrid,
                  "knn": knn_paramgrid,
                  "gaussian": gnb_paramgrid,
                  "perceptron": per_paramgrid}



## reading in files and data wrangling
args = get_args()
top_models = pd.read_csv(args.best_models_in, sep="\t")
meta = pd.read_csv(args.metadata, sep="\t")
ml_data_df = pd.read_csv(args.otu_table, sep='\t')
meta_keys = pd.read_csv(args.metadata_dict, sep='\t')


## inverse dictionary for y-pred column so I can map the final results back to it
inverse_dict = dict(zip(meta_keys["assigned_num"], meta_keys[args.predict_col]))


## getting a list of the top models
int_top_models = top_models.loc[top_models['num_occurences'] >= 3]
top_model_list = int_top_models['model'].to_list()

## filtering model dict to only include models in top_model_list
filt_dict = {}
for value in top_model_list:
    int_dict = {k:v for (k,v) in model_dict.items() if value in k}
    filt_dict.update(int_dict)




## x - the side that has the data I want the model to use to predict y
## y - what is to be predicted
xy_results = make_xy_tables(meta_df=meta,
                            otu_df=ml_data_df,
                            merge_on=args.sample_id,
                            y_col=args.predict_col)

x_dataframe = xy_results["x_dataframe"]
y_dataframe = xy_results["y_dataframe"]



## now I can actually run my top models through the voter classifier
av_correct_list = []
comb_meta_dict = {}

for label, model_func in filt_dict.items():
    ## saving the name of the model I'm running
    model_label = label
    ## generating a new dictionary of models for the estimator list
    new_dict = {k:v for (k,v) in filt_dict.items() if model_label not in k}
    
    ## putting together estimator list
    estimator_list = []
    for label, model in new_dict.items():
        model_tuple = (label, model)
        estimator_list.append(model_tuple)
    
    ## using the model name to pull the correct hyperparameters out of the dictionary
    wanted_params = paramgrid_dict[model_label]

    print(model_func)

    ## now we have all the parts that we need so we can actually run this function!!
    predict_results = kfold_model_predict(x_dataframe=x_dataframe,
                                          y_dataframe=y_dataframe,
                                          k_fold=kf,
                                          wanted_model=model_func,
                                          paramgrid=wanted_params,
                                          wanted_model_name=model_label,
                                          add_estimator_list=estimator_list)
    
    ## pulling model results
    ## average amount of times that the model predicts the correct y-value
    model_acc = predict_results["model_mean_acc"]
    overall_model_results = pd.DataFrame(data=model_acc,
                                        index=["av_correct"]).T
    final_model_acc = overall_model_results["av_correct"].mean()

    av_correct_list.append((model_label, final_model_acc))

    ## joining y-pred key and values to the metadata file
    pre_y_pred = predict_results["model_y_pred"]
    y_pred = pd.concat(pre_y_pred, ignore_index=True)
    y_pred["mouse_id"] = y_pred.index
    comb_y_pred = y_pred.merge(meta, how='left', on=["mouse_id"])
    comb_y_pred["model"] = model_label

    comb_meta_dict.update({model_label: comb_y_pred})


## putting together final outputs
selected_model_results = pd.DataFrame(av_correct_list,
                                      columns=["model", "overall_score"])
comb_meta_yPred = pd.concat(comb_meta_dict, ignore_index=True)

## mapping y-pred column values to the inverse dictionary so they're not numeric anymore
comb_meta_yPred["key"] = comb_meta_yPred["key"].map(inverse_dict)
comb_meta_yPred["final_y_pred"] = comb_meta_yPred["final_y_pred"].map(inverse_dict)


## saving my outputs
selected_model_results.to_csv(args.model_res_out, sep="\t")
comb_meta_yPred.to_csv(args.comb_yPreds_out, sep="\t")