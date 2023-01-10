from model_testing import functions
from sklearn.naive_bayes import BernoulliNB


x,y=functions.openDataset()
X_train, X_test, y_train, y_test = functions.splitDataset(x,y)
X_trainv, X_testv = functions.Vectorizer_Cv(X_train,X_test)


#Training the model
BNB=BernoulliNB()
BNB.fit(X_train,y_train)
NB_score = BNB.score(X_test, y_test)
print("Results for Bernoulli Naive Bayes with count vect, Accuracy= ")
print(NB_score)

y_pred = BNB.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)