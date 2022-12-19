import pandas as pd
import random

dataB=pd.read_csv('../datafile/IMDB Dataset.csv')
dataA=pd.read_csv('datafile/preproc_data.csv')
for j in range(0,10):
    i=random.randint(0,dataA.shape[0])
    print('Review',i, '\n--------------------------- Raw data: --------------------------------')
    print(dataB.iloc[i]['review'])
    print('                                      Clean data:                                    ')
    print(dataA.iloc[i]['review'],'\n')