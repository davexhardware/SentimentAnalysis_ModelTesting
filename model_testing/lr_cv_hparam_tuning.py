from sklearn.linear_model import LogisticRegression
from model_testing import functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from skopt import BayesSearchCV
N_FOLDS = 10
x,y=functions.openDataset()
y_data=y.tolist()

cv = CountVectorizer(min_df=2, ngram_range=(1, 2))
X = cv.fit_transform(x)
min_max_scaler = preprocessing.MaxAbsScaler() #scaling the vectorized dataset
x_data= min_max_scaler.fit_transform(X)

def objective(hyparam,x,y,n_folds=N_FOLDS):
    """"returns the validation score for the given hyperparameters"""
    lr=LogisticRegression(penalty="l2",C =hyparam["C"], solver=hyparam["solver"])
    cv_score=cross_validate(lr,X=x,y=y,cv=n_folds,scoring='f1')
    best_score=max(cv_score)
    return {'best':best_score,'params':hyparam}


opt = BayesSearchCV(
    LogisticRegression(max_iter=500,penalty="l2"),
    {
        'C': [5,5.2,5.4,5.6,5.8,6],
        'solver': ["sag"],
    },
    n_iter=5,
    cv=5
)

opt.fit(x_data,y_data)

print("val. score: %s" % opt.best_score_)
print("Opt params: %s" % opt.best_params_)