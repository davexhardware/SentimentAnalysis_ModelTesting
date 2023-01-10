from sklearn import svm
from model_testing import functions

x,y=functions.openDataset()
X_train, X_test, y_train, y_test = functions.splitDataset(x,y)
X_trainv, X_testv = functions.Vectorizer_Cv(X_train,X_test)



#Training the model
svcl = svm.SVC()
svcl.fit(X_train, y_train)
svcl_score = svcl.score(X_test_dtm, y_test)
print("Results for Support Vector Machine with CountVectorizer")
print(svcl_score)

y_pred = svcl.predict(X_test_dtm)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)