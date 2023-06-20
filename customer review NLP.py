# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 00:31:22 2023

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\Data Science\Daily Practice\April\25-05-2023\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review= review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test,y_test)
variance
