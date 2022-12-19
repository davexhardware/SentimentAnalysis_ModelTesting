from model_testing import functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mt
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = functions.openDataset()
X_trainv, X_testv = functions.Vectorizer_Tfidf(X_train,X_test)

#Training the model
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
RF_score = rfc.score(X_test, y_test)
print("Results for Random Forest with TfIdf vect, Accuracy= ")
print(RF_score)

y_pred = rfc.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)