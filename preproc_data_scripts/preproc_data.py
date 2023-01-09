import nltk.tokenize as tok
import spacy as sp
import re
import pandas as pd


# import random
# nltk.download('punkt')
class Preprocessing:
    def createstopw(self):
        stopwords = []
        with open('../datafile/StopWords_Geographic.txt', 'r') as f:
            sw_g = f.readlines()
        with open('../datafile/StopWords_DatesandNumbers.txt', 'r') as f:
            sw_d = f.readlines()
        [stopwords.append(sw.strip('\n').lower()) for sw in sw_g]
        [stopwords.append(sw.strip('\n').lower()) for sw in sw_d]

        return stopwords

    def text_preproc(self, tokrev):
        if not self.stopw:
            self.stopw = self.createstopw()
        tokrev = re.sub(r"[^A-Za-z]+", " ", tokrev)
        renot = re.compile("|".join(map(re.escape, self.notwords)))
        tokrev = renot.sub("not", tokrev)
        tokrev = tok.word_tokenize(tokrev, "english")  # tokenizzo la prima review
        tokrev = [token.lemma_ for token in self.nlp(str(tokrev)) if
                  (not token.is_punct)]  # pulisco dalla punteggiatura
        tokrev = [word for word in tokrev if (
                word != '' and word != ' ' and word != "\'s" and word != 'br' and word != 'em' and word not in self.stopw and len(
            word) > 1)]
        stoken = " ".join(tokrev)
        return stoken

    def preproc_set(self):

        self.dataset = pd.read_csv('../datafile/IMDB Dataset.csv')  # .head(300)
        for i in range(0, self.dataset.shape[0]):
            self.dataset.iloc[i]['review'] = self.text_preproc(self.dataset.iloc[i]['review'].lower())
            if (i % 100) == 0:
                perc = (i / self.dataset.shape[0]) * 100
                if perc > 0.5:
                    print('Percentuale=', str(perc))
        self.dataset.to_csv('../datafile/preproc_data_def.csv', index=False)

    notwords = [
        "nor",
        "don t",
        "won t",
        "couldn",
        "didn",
        "aren",
        "doesn",
        "hasn",
        "hadn",
        "haven",
        "isn",
        'mightn',
        'mustn',
        'needn',
        'shan',
        'shouldn',
        'wasn',
        'weren',
        'wouldn',
    ]

    def __init__(self):
        self.nlp = sp.load("en_core_web_md")
        self.stopw = self.createstopw()
