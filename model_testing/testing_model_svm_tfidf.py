from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import sklearn.metrics as mt
from matplotlib import pyplot as plt
from model_testing import functions

x,y=functions.openDataset()
X_train, X_test, y_train, y_test = functions.splitDataset(x,y)
X_trainv, X_testv = functions.Vectorizer_Tfidf(X_train,X_test)


#Training the model
svcl = svm.SVC()
svcl.fit(X_train, y_train)
svcl_score = svcl.score(X_test, y_test)
print("Results for Support Vector Machine with tfidf")
print(svcl_score)

y_pred = svcl.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)