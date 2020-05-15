# Adapted from: https://www.kaggle.com/sudalairajkumar/xgb-starter-in-python
import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from preprocess import preprocess_data_1, preprocess_data_2
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# df = pd.read_json(open("train.json", "r"))

# df["num_photos"] = df["photos"].apply(len)
# df["num_features"] = df["features"].apply(len)
# df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
# df["created"] = pd.to_datetime(df["created"])
# df["created_month"] = df["created"].dt.month
# df["created_day"] = df["created"].dt.day

# num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
#              "num_photos", "num_features", "num_description_words",
#              "created_month", "created_day"]

# # Encode categorical features
# categorical = ["display_address", "manager_id", "building_id", "street_address"]
# for f in categorical:
#         if df[f].dtype=='object':
#             #print(f)
#             lbl = preprocessing.LabelEncoder()
#             lbl.fit(list(df[f].values))
#             df[f] = lbl.transform(list(df[f].values))
#             num_feats.append(f)


# # We have features column which is a list of string values. 
# # So we can first combine all the strings together to get a single string and then apply count vectorizer on top of it.
# df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
# tfidf = CountVectorizer(stop_words='english', max_features=100)
# tr_sparse = tfidf.fit_transform(df["features"])

# train_X = sparse.hstack([df[num_feats], tr_sparse]).tocsr()

# target_num_map = {'high':0, 'medium':1, 'low':2}
# train_y = np.array(df['interest_level'].apply(lambda x: target_num_map[x]))

# print('Total data shape:', train_X.shape)
# split = int(train_X.shape[0]/3)
# train_X = train_X[:-split]
# train_y = train_y[:-split]
# test_X = train_X[-split:]
# test_y = train_y[-split:]
# print('Train data length:', train_X.shape[0])
# print('Test data length:', test_X.shape[0])

train_X, test_X, train_y, test_y = preprocess_data_1()
train_X = train_X.values
test_X = test_X.values
train_y = train_y.values
test_y = test_y.values
le = preprocessing.LabelEncoder()
train_y = le.fit_transform(train_y)
test_y = le.transform(test_y)



def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break

# Train
# y_train_pred, model = runXGB(train_X, train_y, train_X, train_y, num_rounds=400)
# Test
y_test_pred, model = runXGB(train_X, train_y, test_X, num_rounds=400)

# print("train log loss:", log_loss(train_y, y_train_pred))
# print("train acc:", accuracy_score(train_y, np.argmax(y_train_pred, axis = 1)))
print("test log loss:", log_loss(test_y, y_test_pred))
print("test acc:", accuracy_score(test_y, np.argmax(y_test_pred, axis = 1)))