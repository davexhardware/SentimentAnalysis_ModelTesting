from sklearn.linear_model import LogisticRegression
from model_testing import functions

X_train, X_test, y_train, y_test = functions.openDataset()
X_trainv, X_testv = functions.Vectorizer_Tfidf(X_train,X_test)


#Training the model
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr_score = lr.score(X_test, y_test)
print("Results for Logistic Regression with tfidf")
print(lr_score)

y_pred= lr.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)
