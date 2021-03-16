import re
import os
import sys
import json

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC  #we will use svm model
import re
from sklearn.pipeline import Pipeline
import utils
import joblib

df = pd.read_excel("IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx")
df["Reviews"]=df["Reviews"].apply(lambda x:utils.get_clean(x))

X = df["Reviews"]
y = df["Sentiment"]

x_train,x_test, y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

model = Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('trainer',LinearSVC())
])

model.fit(x_train,y_train)

joblib.dump(model,"model.joblib")

