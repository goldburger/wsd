import csv
import math
from collections import defaultdict
from collections import Counter
from functools import reduce

# Constants
WSD_TRAIN = '../data/wsd_train.txt'
WSD_DEV = '../data/wsd_dev.txt'
WSD_TEST = '../data/wsd_test.txt'

class Bayes:
    def __init__(self, examples):
        self.prior = dict()
        self.likelihood = dict()
        
        # group by class (word sense)
        self.classes_l = [e[0] for e in examples]
        self.classes = set(self.classes_l)
    
        self.bigdoc = defaultdict(list)
        for e in examples:
            self.bigdoc[e[0]] += e[1].split(' ')

    # Functions
    def train(self):
        self.compute_prior()
        self.compute_likelihood()
    
    def compute_prior(self):
        class_count = Counter(self.classes_l)
        self.prior = {c: class_count[c]/len(self.classes_l) for c in self.classes}
        x = 3

    def compute_likelihood(self):
        word_count = {c: Counter(self.bigdoc[c]) for c in self.classes}
        for c in self.classes:
            self.likelihood[c] = {w: word_count[c][w]/len(self.bigdoc[c]) for w in self.bigdoc[c]}

    def log(self, n):
        return math.log(n, 2)

    def predict(self, sentence):
        words = sentence.split(' ')
        best = 0
        prediction = 'None'
    
        for c in self.classes:
            prior_score = self.log(self.prior[c])

            l = lambda w: self.log(self.likelihood[c][w]) if w in self.likelihood[c] else 0
            word_likelihoods = list(map(l, words))
            likelihood_score = sum(word_likelihoods)
            
            score = prior_score + likelihood_score

            if score < best:
                best = score
                prediction = c

        return prediction

    def report_score(self, num_correct, num_total):
        """ 
        This function can be altered or modified in the future to
        utilize more refined reporting methods -- precision, recall,
        f-score, perplexity, etc.
        One might, in fact, create a Score object, which is used to keep tabs
        on various measures during testing.
        """
        print("Predicted word sense with {0}% accuracy\n".format(num_correct/num_total * 100))

    
# Non-module entry point:
if __name__ == '__main__':
    """ EXTRACT THE TRAINING DATA """
    # get the list of ('class', 'training instance') pairs
    with open(WSD_TRAIN, 'r') as f:
        examples = list(csv.reader(f, delimiter='\t'))

    """ CREATE AND TRAIN THE MODEL """
    bayes_classifier = Bayes(examples)
    bayes_classifier.train()

    """ TEST THE MODEL """
    with open(WSD_TEST, 'r') as f:
        tests = list(csv.reader(f, delimiter='\t'))

    num_correct = 0
    for t in tests:
        if bayes_classifier.predict(t[1]) == t[0]:
            num_correct += 1

    bayes_classifier.report_score(num_correct, len(tests))
