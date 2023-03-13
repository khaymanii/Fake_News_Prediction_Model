# Import the Libraries

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download("stopwords")

# printing the stopwords in english

print(stopwords.words('english'))


# Data Collection & Preprocessing

news_dataset = pd.read_csv('train.csv')

news_dataset.shape
news_dataset.head()
news_dataset.isnull().sum()


# replacing the null values with empty string

news_dataset = news_dataset.fillna('')


# merging the author name and news title

news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']


X = news_dataset.drop(columns='label', axis=1)
y = news_dataset['label']

# Stemming

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)


# Seperating the data and label

X = news_dataset['content'].values
y = news_dataset['label'].values


# Converting the textual data to numerical data

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# Training and Testing data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 2)


# Building the Logistic Regression Algorithm model

model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluation using the accuracy score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy score of the test data : ', test_data_accuracy)






