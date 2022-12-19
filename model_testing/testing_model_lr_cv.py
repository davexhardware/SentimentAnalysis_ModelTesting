from sklearn.linear_model import LogisticRegression
from model_testing import functions
from sklearn import preprocessing
#from preproc_data_scripts.preproc_data import Preprocessing

X_train, X_test, y_train, y_test = functions.openDataset()
X_trainv, X_testv = functions.Vectorizer_Cv(X_train,X_test)

min_max_scaler = preprocessing.MaxAbsScaler() #scaling the vectorized dataset
X_tr_s = min_max_scaler.fit_transform(X_trainv)
X_test_s = min_max_scaler.transform(X_testv)
#training the model with logistic regression
lr = LogisticRegression()
lr.fit(X_tr_s, y_train)
lr_score = lr.score(X_test_s, y_test)
print("Results for Logistic Regression with CountVectorizer")
print(lr_score)

y_pred= lr.predict(X_test_s)
tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s = functions.model_parameters(y_test, y_pred)
#exporting the model
file="../export/lr_cv0.pkl"
functions.exportmodel(lr,file)
#try to import the model and predict a new sentiment
"""
lrmodel=functions.importmodel(file)
review="this wasn't a good film"
prep=Preprocessing()
prep_rev=prep.text_preproc(review)
print(prep_rev)
cv=functions.importmodel("../export/cv_save.pkl")
vect_rev=cv.transform([prep_rev])
print(lrmodel.predict(vect_rev),lrmodel.predict_proba(vect_rev))"""