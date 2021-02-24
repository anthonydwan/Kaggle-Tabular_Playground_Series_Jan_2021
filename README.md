# Kaggle-Tabular_Playground_Series_Jan_2021
Predicting a continuous target (regression) based on number of anonymised features columns given in the data. Below is a documentation of the changes and techniques tried/implemented for reference.

**NB:** You may view the notebook in [here](https://nbviewer.jupyter.org/github/anthonydwan/Kaggle-Tabular_Playground_Series_Jan_2021/blob/main/Tabular_Playground_Jan21_%28LGBM_CB_Approach_3_and_4%29.ipynb) (since github only renders static notebooks).


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

### <b>Attempt 4 (Submissions 3 and 4):</b>

Back to CB/LGBM/XGB, kept 2nd order terms, using optuna to tune all three models. 
Results: 
* Optuna XGB never converged lower than 0.702 validation RMSE - was not submitted. 
* Optuna LGBM converged to 0.6987 validation RMSE 
* Optuna CB converged to 0.6985 validation RMSE (there was a small bug which led to suboptimal hyperparameter tuning)
I used average of CB and LGBM for submission 3 and just CB for submission 4. Both did not do better than my initial score. Seems that 2nd order term was not helpful in lifting the score. 

### <b>Attempt 5:</b>
No feature Engineering for 2nd order term, simple optuna tuning on LGBM and CB. 
* Submission 5 - LGBM regressor with optuna hyperparameter tuning 
* Submission 6 - CatBoost regressor with optuna hyperparameter tuning 
* Submission 7 - XGB regressor with optuna hyperparameter tuning 

None of the results were close to satisfactory. 

### <b>Attempt 6:</b> 
I realised my mistake in hyperparameter tuning was over reliance on one held-out validation set as the metric for tuning performance. This led to overfit on that particular set. This should have been obvious since the hyperparameter tuning tool provided by sklearn uses cross-validation on entire training set. 

Thus, I switched the objective metric of optuna tuner to cross_val_score (6-fold CV which I believe to be optimal for 6-core CPU parallel processing) the to ensure that we evaluate a more generalised performance. Given the unsatisfactory training speed after implementing this change, I have tinkered with the pruners in the optuna library. In the hopes of speeding up hyperparameter tuning, I switched to hyperband pruner. 

* Submission 8 & 9 - Optuna + cross_val_score + hyperband pruner LGBM regressor (LB RMSE 0.70182) which is starting to be close to my previous best score
* Submission 10 - Optuna + cross_val_score for average of both LGBM regressor and xgboost regressor


### <b>Attempt 7:</b> 
* Submission 11 - smaller learning rate 
Upon reading the [discussion](https://www.kaggle.com/c/tabular-playground-series-jan-2021/discussion/212520) shown by shogosuzuki, I noticed that the models converge with a significantly better RMSE when a very small learning rate is used. 

I borrowed this approach for optimised LGBM (LB 0.69760)







