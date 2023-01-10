from sklearn import tree
from model_testing import functions

x,y=functions.openDataset()
X_train, X_test, y_train, y_test = functions.splitDataset(x,y)
X_trainv, X_testv = functions.Vectorizer_Tfidf(X_train,X_test)


#Training the model
dtc=tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
DT_score = dtc.score(X_test, y_test)
print("Results for Decision Tree with TfIdf vect, Accuracy= ")
print(DT_score)
tree.plot_tree(dtc)

y_pred= dtc.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)