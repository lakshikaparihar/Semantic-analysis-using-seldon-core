import nltk
from nltk import WordNetLemmatizer
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin

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

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def get_wordcounts(self,x):
	    self.length = len(str(x).split())
	    return self.length

    def get_charcounts(self,x):
    	self.s = x.split()
    	x = ''.join(self.s)
    	return len(x)

    def get_avg_wordlength(self,x):
    	self.count = get_charcounts(x)/get_wordcounts(x)
    	return self.count

    def _get_stopwords_counts(self,x):
    	self.l = len([t for t in x.split() if t in stopwords])
    	return self.l

    def _get_hashtag_counts(self,x):
    	self.l = len([t for t in x.split() if t.startswith('#')])
    	return self.l

    def _get_mentions_counts(self,x):
    	self.l = len([t for t in x.split() if t.startswith('@')])
    	return self.l

    def _get_digit_counts(self,x):
    	self.digits = re.findall(r'[0-9,.]+', x)
    	return len(self.digits)

    def _get_uppercase_counts(self,x):
    	return len([t for t in x.split() if t.isupper()])

    def _cont_exp(self,x):
    	abbreviations = json.load(open("abbereviations_wordlist.json"))

    	if type(x) is str:
    		for key in abbreviations:
    			self.value = abbreviations[key]
    			self.raw_text = r'\b' + key + r'\b'
    			x = re.sub(self.raw_text, self.value, x)
    			# print(raw_text,value, x)
    		return x
    	else:
    		return x


    def get_emails(self,x):
    	self.emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    	self.counts = len(self.emails)

    	return self.counts, self.emails


    def remove_emails(self,x):
    	return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

    def get_urls(self,x):
    	self.urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
    	self.counts = len(self.urls)

    	return self.counts, self.urls

    def remove_urls(self,x):
    	return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

    def remove_rt(self,x):
    	return re.sub(r'\brt\b', '', x).strip()

    def remove_special_chars(self,x):
    	x = re.sub(r'[^\w ]+', "", x)
    	x = ' '.join(x.split())
    	return x

    def remove_html_tags(self,x):
    	return BeautifulSoup(x, 'html.parser').get_text().strip()

    def remove_accented_chars(self,x):
    	x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    	return x

    def remove_stopwords(self,x):
    	return ' '.join([t for t in x.split() if t not in stopwords])	

    def make_base(self,x):
    	x = str(x)
    	self.x_list = []
    	self.doc = nlp(x)
    
    	for token in doc:
    		self.lemma = token.lemma_
    		if self.lemma == '-PRON-' or self.lemma == 'be':
    			self.lemma = token.text

    		self.x_list.append(self.lemma)
    	return ' '.join(self.x_list)

    def get_value_counts(self,df, col):
    	self.text = ' '.join(df[col])
    	self.text = self.text.split()
    	self.freq = pd.Series(self.text).value_counts()
    	return self.freq

    def remove_common_words(self,x, freq, n=20):
    	self.fn = freq[:n]
    	x = ' '.join([t for t in x.split() if t not in self.fn])
    	return x

    def remove_rarewords(self,x, freq, n=20):
    	self.fn = freq.tail(n)
    	x = ' '.join([t for t in x.split() if t not in self.fn])
    	return x

    def remove_dups_char(self,x):
    	x = re.sub("(.)\\1{2,}", "\\1", x)
    	return x

    def spelling_correction(self,x):
    	x = TextBlob(x).correct()
    	return x

    def get_basic_features(self,df):
    	if type(df) == pd.core.frame.DataFrame:
    		df['char_counts'] = df['text'].apply(lambda x: get_charcounts(x))
    		df['word_counts'] = df['text'].apply(lambda x: get_wordcounts(x))
    		df['avg_wordlength'] = df['text'].apply(lambda x: get_avg_wordlength(x))
    		df['stopwords_counts'] = df['text'].apply(lambda x: _get_stopwords_counts(x))
    		df['hashtag_counts'] = df['text'].apply(lambda x: _get_hashtag_counts(x))
    		df['mentions_counts'] = df['text'].apply(lambda x: _get_mentions_counts(x))
    		df['digits_counts'] = df['text'].apply(lambda x: _get_digit_counts(x))
    		df['uppercase_counts'] = df['text'].apply(lambda x: _get_uppercase_counts(x))
    	else:
    		print('ERROR: This function takes only Pandas DataFrame')
    
    	return df


    def get_ngram(self,df, col, ngram_range):
    	self.vectorizer = CountVectorizer(ngram_range=(ngram_range, ngram_range))
    	self.vectorizer.fit_transform(df[col])
    	self.ngram = self.vectorizer.vocabulary_
    	self.ngram = sorted(ngram.items(), key = lambda x: x[1], reverse=True)

    	return self.ngram

    def fit(self, X, y=None):
        return self

    def transform(self,x):
        x = str(x).lower().replace('\\','').replace('_',' ')
        x = self._cont_exp(x)
        x = self.remove_emails(x)
        x = self.remove_urls(x)
        x = self.remove_html_tags(x)
        x = self.remove_rt(x)
        x = self.remove_accented_chars(x)
        x = self.remove_special_chars(x)
        x = re.sub("(.)\\1{2,}", "\\1", x)
        return x

