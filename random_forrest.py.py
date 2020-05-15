# Adapted from: https://www.kaggle.com/aikinogard/random-forest-starter-with-numerical-features
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from preprocess import preprocess_data_1, preprocess_data_2


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
# X = df[num_feats]
# y = df["interest_level"]

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

X_train, X_val, y_train, y_val = preprocess_data_1()
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)

y_train_pred = clf.predict_proba(X_train)
y_val_pred = clf.predict_proba(X_val)
print("train log loss:", log_loss(y_train, y_train_pred))
print("train acc:", accuracy_score(y_train, clf.predict(X_train)))
print("test log loss:", log_loss(y_val, y_val_pred))
print("test acc:", accuracy_score(y_val, clf.predict(X_val)))