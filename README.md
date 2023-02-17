# SentimentAnalysis_ModelTesting
This Repository contains the various scripts used for the Sentimental Analysis project of group "I Sentimentalysti" for the course of Software Engineering @ Politecnico di Bari. We started with a 50k reviews dataset from IMDB (found on Kaggle) and created a model which can be used for a binary classification of user reviews in two categories (positive or negative).
In detail, this is the <b>first of three</b> repositories, and those files contain:
- preprocess algorithms for the dataset (datafile/'IMDB Dataset.csv' --> datafile/preproc_data_def.csv)
- different ML algorithms tested for the Natural Language Processing and classification of reviews (model_testing)
- a comparison between the algorithms's performance (datafile/'tabella risultati.xlsx')
- the exported final model (Logistic Regression with CounVectorizer), the Vectorizer in Pickle format

The other repositories are:<br>
Implementation of the backend server (Uvicorn and FastAPI) that can be used to access to the model implementation via HTML request on local port 5555. The server implements a "/healt_check" access point (GET method only) and a "/prediction" feature (POST method only) which receives in input the sentence and returns the prediction result and the confidence score: <a href="https://github.com/davexhardware/SentimentAnalysis_Backend">Backend repo</a><br>
Last repository: implementation of the website with a landing page used to send Reviews and receive back the results of the predictions, from the backend server. (Angular Material needed):  <a href="https://github.com/davexhardware/SentimentAnalysis_Frontend">Frontend</a>
