import csv
import math
from collections import defaultdict
from collections import Counter

# Constants
WSD_TRAIN = '../data/wsd_train.txt'
WSD_DEV = '../data/wsd_dev.txt'
WSD_TEST = '../data/wsd_test.txt'

# Global variables
class_s = set()
classes = class_s

class_prob = dict()
prior = class_prob
word_prob = dict()
likelihood = word_prob

bigdoc = dict()


# Functions
def compute_prior():
    class_count = Counter(classes_l)
    class_prob = {c: class_count[c]/len(classes_l) for c in classes_s}

def compute_likelihood():
    word_count = {c: Counter(bigdoc[c]) for c in classes_s}
    word_prob = dict()
    for c in classes_s:
        word_prob[c] = {w: word_count[c][w]/len(bigdoc[c]) for w in bigdoc[c]}

def log(n):
    math.log(n, 2)

def predict(sentence):
    words = sentence.split(' ')
    best = 0
    prediction = 'None'
    
    for c in classes:
        prior_score = log(prior[c])
        likelihood_score = reduce(lambda w, a: a + log(likelihood[c][w]), words)
        score = prior_score + likelihood_score

        if score > best:
            best = score
            prediction = c

def report_score(num_correct, num_total):
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

    # group by class (word sense)
    classes_l = [e[0] for e in examples]
    classes_s = set(classes_l)
    
    bigdoc = defaultdict(list)
    for e in examples:
        bigdoc[e[0]] += e[1].split(' ')

    # NB: because the example sentences have been carefully prepared,
    # it is sufficient to split on white space, as we just have

    """ TRAIN THE MODEL """
    # compute class probabilities
    compute_prior()
        
    # compute word probabilities by class
    compute_likelihood()

    """ TEST THE MODEL """
    with open(WSD_TEST, 'r') as f:
        tests = list(csv.reader(f, delimiter='\t'))

    num_correct = 0
    for t in tests:
        if predict(t[1]) == t[0]:
            num_correct += 1

    report_score(num_correct, len(tests))
