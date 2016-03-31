# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import sklearn
import numpy as np

# Load the data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

# Prepare the inputs for the model
train_y = train_df['ACTION']
train_X = train_df.drop(['ACTION'], axis = 1)

# Do five fold cross validation over different parameters
clf=xgb.sklearn.XGBClassifier()
#Choose all predictors except target & IDcols

# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, train, predictors)

# use a full grid over all parameters
param_grid = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)

}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
print("starting GridSearchCV")
grid_search.fit(train_X, train_y)

report(grid_search.grid_scores_)


test_X = test_df.drop(['id'], axis = 1)

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

#Output our predictions in the correct csv format
submission = pd.DataFrame({'Action': predictions, 'Id': test_df['id']})
submission = submission(['id'],['Action'])
submission.to_csv("hw3_2_submission.csv", index=False)