from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle
import sklearn.metrics as mt


dataname='../datafile/preproc_data_def.csv'
def openDataset():
    dataset = pd.read_csv(dataname)
    x = dataset['review']
    y = dataset['sentiment']
    return train_test_split(x, y, test_size=0.2, random_state=24)


def Vectorizer_Tfidf(X_train, X_test,maxdf):
    vector = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    X_train = vector.fit_transform(X_train)
    X_test = vector.transform(X_test)
    exportmodel(vector,"../export/tfidf_save.pkl")
    return X_train, X_test


def Vectorizer_Cv(X_train, X_test):
    cv = CountVectorizer(min_df=2, ngram_range=(1, 2))
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)
    exportmodel(cv,"../export/cv_save.pkl")
    return X_train, X_test

def model_parameters(y_test, y_pred):
    cm_nb = mt.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    cm_nb.figure_.suptitle("Confusion Matrix")
    plt.show()
    tn, fp, fn, tp = cm_nb.confusion_matrix.ravel()
    print('True Positive, False Positives, True Negatives, False Negatives')
    print(tp, fp, tn, fn)
    prec = round(tp / (tp + fp), 4)
    print('Precision: ', prec)  # High Precision: identify as positive cases only the real positives
    # True positive (or Recall) and true negative rates:
    # Recall: The predicted positive  labels /on the total of real positive cases (or negative)
    tpr_lr = round(tp / (tp + fn), 4)  # High recall: spotting most of positive cases in the set
    tnr_lr = round(tn / (tn + fp), 4)
    print('True positive (Recall)/ true negative rates: ', tpr_lr, tnr_lr)
    # F1 Score: Harmonic Mean of Recall and Precision (reciprocal of the mean  on reciprocal values) n=2
    F1s = 2 * (tpr_lr * prec) / (tpr_lr + prec)
    print('F1 Score: ', F1s)
    return tn, fp, fn, tp, prec, tpr_lr, tnr_lr, F1s
def exportmodel(model, filename):
    with open(filename,'wb') as file:
        pickle.dump(model,file)

def importmodel(filename):
    with open(filename,'rb') as file:
        model=pickle.load(file)
    return model
