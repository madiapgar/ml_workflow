# Basic machine learning workflow for scientists and researchers

This repository contains a machine learning workflow that can be used to predict trends in scientific data using the python package `scikit-learn`. The data used for this workflow is associated with the research project in [this](https://github.com/madiapgar/diet_mouse_cdiff) repository. 

## Workflow details:

**Overview**
- Takes input data, metadata, and column name of what you want to be predicted and finds which machine learning models performed the best with the data before running a voter classifier using those top-performing models to give the final predictions and accuracy scores. 
- Also gives the option to run Boruta SHAP on your data to identify important features (default is True). 
- This workflow employs K-Fold cross validation (n_splits=5).

**Inputs**

- The desired project name and location 

- **Files (and paths to them):**
    - Metadata file (machine learning approved)
    - File containing encoded keys for the metadata file
    - File with data you want to use to predict the given metadata variable (wide format)

- **Column names:**
    - The name of the *sample ID* column (this is the name of the column of identifying information that links the metadata and data files)
    - The name of the metadata column that you want to predict (y-pred)

**Outputs**

- **Base:**
    - `best_performing_models.tsv`: gives you the models that consistently had performance scores of 95% or above 
    - `comb_meta_yPred.tsv`: gives you the final predictions combined with the metadata file for easy comparison 
    - `model_predict_scores.tsv`: gives you how well the top models selected performed in their predictions overall

- **If you also ran Boruta SHAP:**
    - `borutaShap_train_feat.tsv`: the top features in the training dataset that were important for what you want predicted
    - `shap_plot.pdf`: a summary SHAP plot of the most important features to your model and why

**Models used**

- Logistic Regression
- Support Vector Classification (SVC)
- KNeighbors Classifier (KNN)
- Gaussian Naive-Bayes Classifier 
- Perceptron
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Ridge Classifier CV 







