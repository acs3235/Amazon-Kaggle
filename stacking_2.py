import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier

# Load the data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)


# Prepare the inputs for the model
train_y = train_df['ACTION']
train_X = train_df.drop(['ACTION'], axis = 1)
test_X = test_df.drop(['id'], axis = 1)


#Split the model inputs into two folds
fold1_y, fold2_y, fold1_X,fold2_X = train_test_split(train_y,train_X, test_size = 0.5)

#INPUT CLASSIFIERS HERE!!

clf1 = xgb.XGBClassifier(max_depth=18, colsample_bytree=0.45, n_estimators = 99)
clf2 = AdaBoostClassifier()
clf3 = LogisticRegression(penalty='l2',C=3)

#Make an encoder for use with logistic regression
encoder = preprocessing.OneHotEncoder()
encoder.fit(np.vstack((train_X,test_X)))


#Use the first fold to generate predictions for the second fold for a set of models
fold1_X_encoded = encoder.transform(fold1_X)
fold2_X_encoded = encoder.transform(fold2_X)

#model1
model1 = clf1.fit(fold1_X, fold1_y)
model1_p = model1.predict_proba(fold2_X)[:, 1]
#model2
model2 = clf2.fit(fold1_X_encoded, fold1_y)
model2_p = model2.predict_proba(fold2_X_encoded)[:, 1]
#model3
model3 = clf3.fit(fold1_X_encoded, fold1_y)
model3_p = model3.predict_proba(fold2_X_encoded)[:, 1]

#Use logisitic regression to determine the weights of these models based on their accuracy with the fold2 predictions
predictions = pd.DataFrame({'1': model1_p, '2': model2_p, '3':model3_p})
stack_model = LogisticRegression().fit(predictions, fold2_y)

#Train these classifiers on the whole train data set
train_X_encoded = encoder.transform(train_X)
m1 = clf1.fit(train_X,train_y)
m2 = clf2.fit(train_X_encoded,train_y)
m3 = clf3.fit(train_X_encoded,train_y)

#Apply these models to the test data
test_X_encoded = encoder.transform(test_X)
m1p = m1.predict_proba(test_X)[:, 1]
m2p = m2.predict_proba(test_X_encoded)[:, 1]
m3p = m3.predict_proba(test_X_encoded)[:, 1]

#Apply the model to our predictions
predictions_2 = pd.DataFrame({'1': m1p, '2': m2p, '3':m3p})
combined_predictions = stack_model.predict_proba(predictions_2)[:, 1]

submission = pd.DataFrame({'Action': combined_predictions, 'Id': test_df['id']},columns = ['Id','Action'])
submission.to_csv("stacking_probs_3.csv", index=False)