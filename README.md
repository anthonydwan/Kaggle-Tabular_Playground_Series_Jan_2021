# Kaggle-Tabular_Playground_Series_Jan_2021
Predicting a continuous target based on number of features columns given in the data. Below is a documentation of the changes and techniques tried/implemented for reference.


### <b>Attempt 1 (Submission 1): </b>

Feature Scaling - MinMax/StandardScaler --> Popular regression models (trial and error), hyperparameter tuning with sklearn GridSearchCV 
1. Linear Regression
2. Support Vector Regression
3. sklearn Decision Tree
4. sklearn Random Forest 
5. CatBoost Regression
6. Lightgbm Regression
7. XGB Regression
8. Averaging all 7 moels of the above. But models with lesser performance dragged down the predictive power of the other models. 

The highest performing model cb regression was used for submission 1 (RMSE at 0.701 (Public)). 

### <b>Attempt 2 (Submission 2): </b>

Feature Engineering - Creating Interaction features (multiply each other to another for 2nd order). 
Feature Selection - SelectKBest was used but there was noticable performance decrease. Thus, no selection was used. 
CatBoost Regression, Lightgbm Regression and XGB Regression together (the three highest performing models) - average out the three. 

This did not perform better than submission 1. 

### <b>Attempt 3:</b>

After adding 2nd order feature, used Gaussian Rank Transformation on all features (since features with normal distrbution helps non-tree models). Then, using kerastuner (Hyperband) to tune a NN with 2 to 20 layers (relu Dense layers and dropout regularisation). The model did not perform well at ~0.707

*Note - Adversial Validation was performed and confirmed that train data is good representation of test data (AUC 0.5)*

### <b>Attempt 4:</b>

Back to CB/LGBT/XGB, kept 2nd order terms, using optuna to tune all three models. 
