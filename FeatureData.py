import csv
import re
import os
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

class FeatureData():
    def __init__(self, article_file_path, stances_file_path):
        self.number_of_classes = 4
        self.classes = ["Agrees", "Disagrees", "Discusses", "Unrelated"]
        self.articles = self._get_articles(article_file_path) # list of dictionaries
        self.stances = self._get_stances(stances_file_path)
        self.number_of_stances = len(self.stances)
        self.number_of_articles = len(self.articles)

    def _get_articles(self, path):
        # Body ID, articleBody
        articles = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                articles.append(row)
        return articles

    def _get_stances(self, path):
        # Headline, Body ID, Stance
        stances = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                stances.append(row)
        return stances

    @staticmethod
    def preprocess_data(sentence):
        text = " ".join(re.findall(r'w+', sentence, flags=re.UNICODE).lower())
        text = [word for word in text if word not in feature_extraction.text.ENGLISH_STOP_WORDS]
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer(token).lower() for token in word_tokenize(text)]

    @staticmethod
    def load_features(feature_func, headlines, bodies, feature_file):
        if not os.path.isfile(feature_file):
            features = feature_func(headlines, bodies)
            np.save(feature_file, features)

        return np.load(feature_file)

fd = FeatureData('./train_bodies.csv', './train_stances.csv')

