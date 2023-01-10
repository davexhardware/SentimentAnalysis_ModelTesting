from sklearn.linear_model import LogisticRegression
from model_testing import functions
from sklearn import preprocessing

x,y=functions.openDataset()
X_train, X_test, y_train, y_test = functions.splitDataset(x,y)
mdf=0.83 # max_df parameter
X_trainv, X_testv = functions.Vectorizer_Cv(X_train,X_test,mdf)
min_max_scaler = preprocessing.MaxAbsScaler() #scaling the vectorized dataset
X_tr_s = min_max_scaler.fit_transform(X_trainv)
X_test_s = min_max_scaler.transform(X_testv)
#training the model with logistic regression
lr = LogisticRegression() #incrementiamo max_iter perché altrimenti il modello non converge
lr.fit(X_tr_s, y_train)
lr_score = lr.score(X_test_s, y_test)
print("Results for Logistic Regression with CountVectorizer and max_df=",mdf)
print("Accuracy: ",lr_score)
y_pred= lr.predict(X_test_s)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)
#exporting the model
file="../export/lr_cv0.pkl"
functions.exportmodel(lr,file)