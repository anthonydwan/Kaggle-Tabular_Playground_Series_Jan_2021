# Kaggle-Tabular_Playground_Series_Jan_2021
predicting a continuous target based on number of features columns given in the data. 


Attempt 1 (Submission 1): 

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

Attempt 2 (Submission 2):

Feature Engineering - Creating Interaction features (multiply each other to another for 2nd order). 
Feature Selection - SelectKBest was used but there was noticable performance decrease. Thus, no selection was used. 
CatBoost Regression, Lightgbm Regression and XGB Regression together (the three highest performing models) - average out the three. 

This did not perform better than submission 1. 

Attempt 3:

After adding 2nd order feature, used Gaussian Rank Transformation on all features (since features with normal distrbution helps non-tree models). Then, using kerastuner (Hyperband) to tune a NN with 2 to 20 layers (relu Dense layers and dropout regularisation). The model did not perform well at ~0.707

Attempt 4:

Back to CB/LGBT/XGB, kept 2nd order terms, using optuna to tune all three models. 
