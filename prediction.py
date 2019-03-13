from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import pandas as pd
import string
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pickle


with open("tweets.pkl", "rb") as file:
    X = pickle.load(file)

with open("output.pkl", "rb") as file:
    y = pickle.load(file)

select = SelectFromModel(LogisticRegression(class_weight='balanced',penalty="l1",C=0.01))
X_ = select.fit_transform(X,y)

x_tr, x_te, y_tr, y_te = train_test_split(X_, y)
model = LinearSVC(class_weight='balanced',C=0.01, penalty='l2', loss='squared_hinge',multi_class='ovr')
model.fit(x_tr, y_tr)
y_preds = model.predict(x_te)
report = classification_report( y_te, y_preds )
print("SVM:")
print(report)
print(np.mean(y_te == y_preds))

print("RandomForest")
model = RandomForestClassifier(criterion='gini')
model.fit(x_tr, y_tr)
y_preds = model.predict(x_te)
report = classification_report( y_te, y_preds )
print(report)
print(np.mean(y_te == y_preds))