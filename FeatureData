import csv

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

def main():
    data_set = FeatureData()

main()
