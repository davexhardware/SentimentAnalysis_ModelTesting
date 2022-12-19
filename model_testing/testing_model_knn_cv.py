from sklearn.neighbors import KNeighborsClassifier
from model_testing import functions

X_train, X_test, y_train, y_test = functions.openDataset()
X_trainv, X_testv = functions.Vectorizer_Cv(X_train,X_test)

#Training the model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
print("Results for KNN Classifier with CountVectorizer")
print(knn_score)

y_pred= knn.predict(X_test)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)