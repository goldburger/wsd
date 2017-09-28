from __future__ import division

import csv
import numpy
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from collections import defaultdict

# Constant declarations
WSD_TRAIN = 'data/wsd_train.txt'
WSD_DEV = 'data/wsd_dev.txt'
WSD_TEST = 'data/wsd_test.txt'


class Bayes2:
    # Fields:
    #    classes, prior, theta, vectorizer

    def __init__(self, train_labels, train_texts):
        # Extract the data
        # with open(WSD_TRAIN, 'r') as f:
        #     train_pairs = list(csv.reader(f, delimiter='\t'))
        #     [train_labels, train_texts] = [list(t) for t in zip(*train_pairs)]
        train_pairs = zip(train_labels, train_texts)

        # Compute class data and priors
        self.classes = set(train_labels)
        class_counts = Counter(train_labels)
        self.prior = {c: -log(class_counts[c] / len(train_labels), 2) \
                      for c in self.classes}

        # Divide the training data into classes
        texts = defaultdict(list)
        for p in train_pairs:
            texts[p[0]].append(p[1])
    
        # Vectorize the training data per class
        # (note that these vectorizers expect a document string *in* an iterable)
        self.vectorizer = {c: CountVectorizer() for c in self.classes}
        # 'fv' for feature vector, 'fc' for feature count
        fvs_by_class = {c: self.vectorizer[c].fit_transform(texts[c]).toarray() \
                        for c in self.classes}

        # Compute feature counts per class and likelihoods
        fcs_by_class = {c: numpy.sum(fvs_by_class[c], axis=0) \
                        for c in self.classes}
        doclen = {c: numpy.sum(fcs_by_class[c]) for c in classes}
        
        self.theta = {c: numpy.negative(numpy.log2(fcs_by_class[c] / doclen[c])) \
                      for c in self.classes}

        # Cleanup (to free memory)
        del fvs_by_class


# Classify
def vectorize(bayes, example, clazz):
    if not clazz in bayes.classes:
        raise Exception
    
    if not isinstance(example, str):
        example = " ".join(example)

    return bayes.vectorizer[clazz].transform([example]).toarray()

def predict(bayes, example):
    """ example : numpy.array """
    (max, argmax) = (0, None)

    for c in bayes.classes:
        fv = vectorize(bayes, example, c)
        likelihood = numpy.dot(fv, bayes.theta[c])
        posterior = bayes.prior[c] + likelihood[0]
        
        if posterior > max:
            max = posterior
            argmax = c

    return argmax

def classify(bayes, examples):
    return [predict(bayes, example) for example in examples]
