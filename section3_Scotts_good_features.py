import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array 
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def OneHotEncoder(data, keymap=None):
	if keymap is None:
		keymap = []
		for col in data.T:
			uniques = set(list(col))
			keymap.append(dict((key, i) for i, key in enumerate(uniques)))
	total_pts = data.shape[0]
	outdat = []
	for i, col in enumerate(data.T):
		km = keymap[i]
		num_labels = len(km)
		spmat = sparse.lil_matrix((total_pts, num_labels))
		for j, val in enumerate(col):
			if val in km:
				spmat[j, km[val]] = 1
		outdat.append(spmat)
	outdat = sparse.hstack(outdat).tocsr()
	return outdat, keymap
	
# Load the data
train_df = pd.read_csv('train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

# Prepare the inputs for the model
train_y = np.array(train_df['ACTION'])
train_X = train_df.drop(['ACTION'], axis = 1).as_matrix()
test_X = test_df.drop(['id'], axis = 1).as_matrix()

#Feature Selection
all_data = np.vstack((train_df.ix[:,1:-1], test_df.ix[:,1:-1]))
num_train = np.shape(train_df)[0]
    
# Transform data
dp = group_data(all_data, degree=2) 
dt = group_data(all_data, degree=3)

X = all_data[:num_train]
X_2 = dp[:num_train]
X_3 = dt[:num_train]

X_test = all_data[num_train:]
X_test_2 = dp[num_train:]
X_test_3 = dt[num_train:]

X_train_all = np.hstack((X, X_2, X_3))
X_test_all = np.hstack((X_test, X_test_2, X_test_3))
num_features = X_train_all.shape[1]
#good_features = [0, 7, 8, 10, 12, 20, 24, 33, 36, 37, 38, 41, 42, 43, 47, 53, 63, 64, 67, 69, 71, 75, 82, 85]
good_features = [0, 7, 9, 28, 34, 36, 39, 40, 49, 58, 59, 60, 62, 64, 74]
Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
Xt, keymap = OneHotEncoder(Xt)
X_train = Xt[:num_train]
X_test = Xt[num_train:]

#Split the model inputs into two folds
#fold1_y, fold2_y, fold1_X,fold2_X = train_test_split(train_y, X_train_all, test_size = 0.3)
t, cv = cross_validation.train_test_split(
                                       range(len(train_y)), test_size=.40, 
                                       random_state = 0)
									   
#INPUT CLASSIFIERS HERE!!
clf1 = xgb.XGBClassifier(max_depth=18, colsample_bytree=0.45, n_estimators = 99)
clf2 = AdaBoostClassifier()
clf3 = LogisticRegression(penalty='l2',C=1.485994)

#Make an encoder for use with logistic regression
encoder = preprocessing.OneHotEncoder()
encoder.fit(np.vstack((train_X,test_X)))

#Select data for each model 
fold1_X = train_X[t]
fold2_X = train_X[cv]
fold1_X3 = X_train[t]
fold2_X3 = X_train[cv]
fold1_y = train_y[t]
fold2_y = train_y[cv]

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
model3 = clf3.fit(fold1_X3, fold1_y)
model3_p = model3.predict_proba(fold2_X3)[:, 1]

#Use logisitic regression to determine the weights of these models based on their accuracy with the fold2 predictions
predictions = pd.DataFrame({'1': model1_p, '2': model2_p, '3':model3_p})
stack_model = LogisticRegression().fit(predictions, fold2_y)

#Train these classifiers on the whole train data set
train_X_encoded = encoder.transform(train_X)
m1 = clf1.fit(train_X,train_y)
m2 = clf2.fit(train_X_encoded,train_y)
#m3 = clf3.fit(train_X_encoded,train_y)
m3 = clf3.fit(X_train, train_y)

#Apply these models to the test data
test_X_encoded = encoder.transform(test_X)
m1p = m1.predict_proba(test_X)[:, 1]
m2p = m2.predict_proba(test_X_encoded)[:, 1]
m3p = m3.predict_proba(X_test)[:, 1]

#Apply the model to our predictions
predictions_2 = pd.DataFrame({'1': m1p, '2': m2p, '3':m3p})
combined_predictions = stack_model.predict_proba(predictions_2)[:, 1]

submission = pd.DataFrame({'Action': combined_predictions, 'Id': test_df['id']},columns = ['Id','Action'])
submission.to_csv("stacking_Scotts_features.csv", index=False)