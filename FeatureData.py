import csv
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction

class FeatureData():
    def __init__(self):
        number_of_classes = 4
        classes = ["Agrees", "Disagrees", "Discusses", "Unrelated"]
        articles = self._get_articles() # list of dictionaries
        stances = self._get_stances()
        number_of_stances = len(stances)
        number_of_articles = len(articles)

    def _get_articles(self):
        # Body ID, articleBody
        articles = []
        with open('fnc-1-baseline/fnc-1/train_bodies.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                articles.append(row)
        return articles

    def _get_stances(self):
        # Headline, Body ID, Stance
        stances = []
        with open('fnc-1-baseline/fnc-1/train_stances.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                stances.append(row)
        return stances

    @staticmethod
    def preprocess_data(self, text):
        text = " ".join(re.findall(r'w+', text, flags=re.UNICODE).lower())
        text = [word for word in text if word not in feature_extraction.text.ENGLISH_STOP_WORDS]
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer(token).lower() for token in word_tokenize(text)]

