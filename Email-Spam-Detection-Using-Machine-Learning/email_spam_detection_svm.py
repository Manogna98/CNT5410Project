"""
Description:
              Email spam detection system is used to detect email spam using Machine Learning model and Python,
              where we have a dataset containing a lot of emails from which we extract important words using support vector machine to detect spam email.
"""

## Importing modules ##
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import pickle
from sklearn import svm
nltk.download('stopwords')

dataset = pd.read_csv('Dataset/emails_domain.csv')

# 1.4 Cleaning data from punctuation and stopwords and then tokenizing it into words (tokens)
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

print(f"Dataset head after cleaning punctuation and stopwords and then tokenizing it into words : \n{dataset['text'].head().apply(process)}\n")
print(len(dataset['text']))

# 1.5 Convert the text into a matrix of token counts
message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])
pickle.dump(message, open("Model/vector.pickel", "wb"))    # Save the vectorizer output message
message = pickle.load(open("Model/vector.pickel", "rb"))    # Load the vectorizer output message

# 1.6 Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], train_size=0.80, random_state=0)

## 2 | Model Creation ##
"""Create model to fit it to the data"""

# 2.1 Create and train the Support Vector Machine Model
classifier = svm.SVC().fit(X_train, y_train)

# 2.2 Classifiers prediction
y_pred = classifier.predict(X_test)
print(f"Prediction results (y_pred): \n{y_pred}\n")

## 3 | Model Evaluation ##
"""Evaluate model performance"""
print(f"Classification report :\n{classification_report(y_test, y_pred)}\n")
print(f"Confusion Matrix :\n{confusion_matrix(y_test, y_pred)}\n")
print(f"Model accuracy : {round(accuracy_score(y_test, y_pred), 2)*100} %")

## 4 | Write the spam e-mail IDs to haraka config file for the next steps of the project

## 4.1 Extract the spam e-mail IDs from prediction results
domain_names = dataset['domain'].tolist()
print(len(domain_names))
results = []
domains = []
for i in range(len(y_pred)):
  if(y_pred[i] == 1):
    results.append(domain_names[i])
for i in range(len(results)):
    domains.append(results[i].split('@')[1]) 

results = list(set(results))
domains = list(set(domains))

## 4.2 Write to the Haraka block_me plugin config files to block the e-mails ids
with open('../Haraka/config/host_list', 'w+') as f:
    for i in range(len(domains)):
        f.write(domains[i])
        f.write("\n")
with open('../Haraka/config/host_list.anti_spoof', 'w+') as f:
    for i in range(len(domains)):
        f.write(domains[i])
        f.write("\n")

