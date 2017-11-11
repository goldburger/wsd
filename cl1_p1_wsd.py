"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""

from __future__ import division

import math
from math import log
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
from scipy.stats import norm
from functools import partial
from nltk import word_tokenize
from nltk import sent_tokenize
from random import shuffle

"""
 read one of train, dev, test subsets 
 
 subset - one of train, dev, test
 
 output is a tuple of three lists
 	labels: one of the 6 possible senses <cord, division, formation, phone, product, text >
 	targets: the index within the text of the token to be disambiguated
 	texts: a list of tokenized and normalized text input (note that there can be multiple sentences)

"""
import nltk 
def read_dataset(subset):
	labels = []
	texts = []
	targets = []
	if subset in ['train', 'dev', 'test']:
		with open('data/wsd_'+subset+'.txt') as inp_hndl:
			for example in inp_hndl:
				label, text = example.strip().split('\t')
				text = nltk.word_tokenize(text.lower().replace('" ','"'))
				if 'line' in text:
					ambig_ix = text.index('line')
				elif 'lines' in text:
					ambig_ix = text.index('lines')
				else:
					ldjal
				targets.append(ambig_ix)
				labels.append(label)
				texts.append(text)
		return (labels, targets, texts)
	else:
		print ('>>>> invalid input !!! <<<<<')

"""
computes f1-score of the classification accuracy

gold_labels - is a list of the gold labels
predicted_labels - is a list of the predicted labels

output is a tuple of the micro averaged score and the macro averaged score

"""
import sklearn.metrics
def eval(gold_labels, predicted_labels):
	return ( sklearn.metrics.f1_score(gold_labels, predicted_labels, average='micro'),
			 sklearn.metrics.f1_score(gold_labels, predicted_labels, average='macro') )


"""
a helper method that takes a list of predictions and writes them to a file (1 prediction per line)
predictions - list of predictions (strings)
file_name - name of the output file
"""
def write_predictions(predictions, file_name):
	with open(file_name, 'w') as outh:
		for p in predictions:
			outh.write(p+'\n')


# The Bayes class goes here
# import csv
# import numpy
# from math import log
# from sklearn.feature_extraction.text import CountVectorizer
# from collections import Counter
# from collections import defaultdict
# from nltk import sent_tokenize

# Constant declarations
WSD_TRAIN = 'data/wsd_train.txt'
WSD_DEV = 'data/wsd_dev.txt'
WSD_TEST = 'data/wsd_test.txt'


class Bayes2:
    # Fields:
    #    classes, prior, theta, vectorizer

    def __init__(self, train_labels, train_texts, wd_len=False, sn_len=False):
        # Extract the data
        # with open(WSD_TRAIN, 'r') as f:
        #     train_pairs = list(csv.reader(f, delimiter='\t'))
        #     [train_labels, train_texts] = [list(t) for t in zip(*train_pairs)]
        tokenized_train_texts = train_texts
        train_texts = [" ".join(text) for text in train_texts]
        train_pairs = zip(train_labels, train_texts)

        # Compute class data and priors
        self.classes = set(train_labels)
        class_counts = Counter(train_labels)
        self.prior = {c: -log(class_counts[c] / len(train_labels), 2) \
                      for c in self.classes}

        # Divide the training data into classes
        texts = defaultdict(list)
        tokenized_texts = defaultdict(list)
        self.sentences = defaultdict(list)
        for (label, text)  in train_pairs:
            texts[label].append(text)
            self.sentences[label] += sent_tokenize(text)
        for (label, text) in zip(train_labels, tokenized_train_texts):
            tokenized_texts[label] += text

        for c in self.classes:
            self.sentences[c] = list(map(lambda s: word_tokenize(s), self.sentences[c]))
            
    
        # Vectorize the training data per class
        # (note that these vectorizers expect a document string *in* an iterable)
        self.vectorizer = {c: CountVectorizer() for c in self.classes}
        # 'fv' for feature vector, 'fc' for feature count
        fvs_by_class = {c: self.vectorizer[c].fit_transform(texts[c]).toarray() \
                        for c in self.classes}

        # Compute feature counts per class and likelihoods
        fcs_by_class = {c: numpy.sum(fvs_by_class[c], axis=0) \
                        for c in self.classes}
        doclen = {c: numpy.sum(fcs_by_class[c]) for c in self.classes}
        
        self.theta = {c: numpy.negative(numpy.log2(fcs_by_class[c] / doclen[c])) \
                      for c in self.classes}


        # Compute likelihoods for extra features, if desired
        self.theta_wd_len = defaultdict(int)
        self.theta_sn_len = defaultdict(int)
        
        if wd_len:
            for c in self.classes:
                word_lens = list(map(lambda w: len(w), tokenized_texts[c]))
                # get mu, std
                self.theta_wd_len[c] = norm.fit(word_lens)

        if sn_len:
            for c in self.classes:
                sn_lens = list(map(lambda s: len(s), self.sentences[c]))
                # get mu, std
                self.theta_sn_len[c] = norm.fit(sn_lens)


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
        likelihood_bow = numpy.dot(fv, bayes.theta[c])[0]
        likelihood_wd_len = -log(norm.pdf(avg_wd_len(example), *bayes.theta_wd_len[c]), 2)
        likelihood_sn_len = -log(norm.pdf(avg_sn_len(example), *bayes.theta_sn_len[c]), 2)
        posterior = bayes.prior[c] + likelihood_bow + likelihood_wd_len + likelihood_sn_len
        
        if posterior > max:
            max = posterior
            argmax = c

    return argmax

def classify(bayes, examples):
    return [predict(bayes, example) for example in examples]

def avg_wd_len(example):
    wd_lens = map(lambda w: len(w), example)
    return sum(wd_lens) / len(example)

def avg_sn_len(example):
    # example is, unfortunately, a single string
    los = sent_tokenize(" ".join(example)) # first break it into sentence strings
    lol = list(map(lambda s: word_tokenize(s), los)) # then break them into lists
    sn_lens = list(map(lambda sn: len(sn), lol))
    return sum(sn_lens) / len(los)
    

def get_bow_naivebayes_training_scores(train_texts, train_labels, dev_texts, dev_labels):
        classifier = Bayes2(train_labels, train_texts, True, True)
        predictions = classify(classifier, train_texts)
        return eval(train_labels, predictions)

"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
        classifier = Bayes2(train_labels, train_texts, False, False)
        predictions = classify(classifier, test_texts)
        return eval(test_labels, predictions)


# Creates bags of words for a set from the original lists of sentence words
def make_bow(texts):
  bow = []
  for text in texts:
    next_bow = defaultdict(float)
    for word in text:
      next_bow[word] += 1.
    bow.append(next_bow)
  return bow

def predict_labels(senses, theta, bow):
  labels = []
  for i in range(0, len(bow)):
    scoring = defaultdict(float)
    for sense in senses:
      for word in bow[i]:
        scoring[sense] += bow[i][word] * theta[sense][word]
    v = list(scoring.values())
    k = list(scoring.keys())
    label = k[v.index(max(v))]
    labels.append(label)
  return labels

"""
Trains a perceptron model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_bow_perceptron_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels):
  """
  **Your final classifier implementation of part 3 goes here**
  """
  senses = set(train_labels)

  # Create beginning structure for weights, with all weights zero
  theta = dict()
  m = dict()
  m_last_updated = dict()
  for sense in senses:
    theta[sense] = defaultdict(float)
    m[sense] = defaultdict(float)
    m_last_updated[sense] = defaultdict(float)

  # Create training and test bags of words collection
  train_bow = make_bow(train_texts)
  test_bow = make_bow(test_texts)

  # Initialize list of indices used for randomizing training instance order
  indices = []
  for i in range(0, len(train_texts)):
    indices.append(i)

  test_results_prev = (0, 0)
  test_results = (0, 0)
  predicted_labels_prev = []
  predicted_labels = []

  # Main perceptron loop
  counter = 0
  while (True):

    # Evaluates accuracy on training set after using whole training set once
    if (counter % len(train_texts) == 0 and counter > 0):
      m_temp = dict()
      theta_temp = dict()
      for sense in senses:
        m_temp[sense] = defaultdict(float)
        theta_temp[sense] = defaultdict(float)
        # Obtain weights from running average before evaluating
        for word in m[sense]:
          m_temp[sense][word] = m[sense][word] + theta[sense][word] * (counter - m_last_updated[sense][word])
          theta_temp[sense][word] = m_temp[sense][word] / counter
      print "Results on training set: " + str(eval(train_labels, predict_labels(senses, theta_temp, train_bow)))

      test_results_prev = test_results
      predicted_labels_prev = predicted_labels
      predicted_labels = predict_labels(senses, theta_temp, test_bow)
      test_results = eval(test_labels, predicted_labels)

      print "Result on test set: " + str(test_results)

      # Stopping condition when previous results exceed current
      # Rolls back to previous results and halts in such a case
      if (test_results_prev[0] > test_results[0]):
        print "Previous test result of " + str(test_results_prev) + " exceeded current; rolling back to previous and stopping."
        print "Final test accuracy: " + str(test_results_prev)
        #write_predictions(predicted_labels_prev, "q3p3.txt")
        return test_results_prev

      shuffle(indices)
    index = indices[counter % len(indices)]

    # Obtain predicted from argmax of scores for each class
    scoring = defaultdict(float)
    for sense in senses:
      for word in train_bow[index]:
        scoring[sense] += train_bow[index][word] * theta[sense][word]
    v = list(scoring.values())
    k = list(scoring.keys())
    yhat = k[v.index(max(v))]
   
    # If prediction is wrong, update weight vector
    correct_label = train_labels[index]
    if (yhat != correct_label):
      for word in train_bow[index]:
        # Updates for scores of predicted class
        m[yhat][word] += theta[yhat][word] * (counter - m_last_updated[yhat][word])
        m_last_updated[yhat][word] = counter
        theta[yhat][word] -= train_bow[index][word]
        #if (counter == 0):
        #  print yhat + "_" + word + ":-" + str(train_bow[index][word])
        m[yhat][word] -= train_bow[index][word]

        # Updates for scores of actual class
        m[correct_label][word] += theta[correct_label][word] * (counter - m_last_updated[correct_label][word])
        m_last_updated[correct_label][word] = counter
        theta[correct_label][word] += train_bow[index][word]
        #if (counter == 0):
        #  print correct_label + "_" + word + ":" + str(train_bow[index][word])
        m[correct_label][word] += train_bow[index][word]

    counter += 1

# Helper function used for one extra feature
def average_word_len(bow):
  len_sum = 0
  numwords = 0
  for word in bow:
    len_sum += len(word)
    numwords += bow[word]
  return float(len_sum)/numwords

# Helper function used for other extra feature
def inverse_sentence_len(bow, maxlen):
  sentence_len = 0
  for word in bow:
    sentence_len += bow[word]
  return maxlen - sentence_len

"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
        classifier = Bayes2(train_labels, train_texts, True, True)
        predictions = classify(classifier, test_texts)
        return eval(test_labels, predictions)

"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):

  extra_1 = True
  extra_2 = True

  senses = set(train_labels)

  # Create beginning structure for weights, with all weights zero
  theta = dict()
  m = dict()
  m_last_updated = dict()
  for sense in senses:
    theta[sense] = defaultdict(int)
    m[sense] = defaultdict(int)
    m_last_updated[sense] = defaultdict(int)

  # Obtains max sentence length for training set and test set, for use in extra feature 2
  train_maxlen = 0
  for sentence in train_texts:
    if len(sentence) > train_maxlen:
      train_maxlen = len(sentence)
  test_maxlen = 0
  for sentence in test_texts:
    if len(sentence) > test_maxlen:
      test_maxlen = len(sentence)

  # Create training and test bags of words collection
  train_bow = make_bow(train_texts)
  # Insertion of extra feature 1, if enabled
  if (extra_1):
    #print "Extra feature 1 enabled."
    for bow in train_bow:
      bow["extra 1"] += average_word_len(bow)
  if (extra_2):
    #print "Extra feature 2 enabled."
    for bow in train_bow:
      bow["extra 2"] += inverse_sentence_len(bow, train_maxlen)
  test_bow = make_bow(test_texts)
  # Insertion of extra feature 2, if enabled
  if (extra_1):
    for bow in test_bow:
      bow["extra 1"] += average_word_len(bow)
  if (extra_2):
    for bow in test_bow:
      bow["extra 2"] += inverse_sentence_len(bow, test_maxlen)

  # Initialize list of indices used for randomizing training instance order
  indices = []
  for i in range(0, len(train_texts)):
    indices.append(i)

  test_results_prev = (0, 0)
  test_results = (0, 0)
  predicted_labels_prev = []
  predicted_labels = []

  # Main perceptron loop
  counter = 0
  while (True):

    # Evaluates accuracy on training set after using whole training set once
    if (counter % len(train_texts) == 0 and counter > 0):
      m_temp = dict()
      theta_temp = dict()
      for sense in senses:
        m_temp[sense] = defaultdict(int)
        theta_temp[sense] = defaultdict(int)
        # Obtain weights from running average before evaluating
        for word in m[sense]:
          m_temp[sense][word] = m[sense][word] + theta[sense][word] * (counter - m_last_updated[sense][word])
          theta_temp[sense][word] = m_temp[sense][word] / counter
      #print "Results on training set: " + str(eval(train_labels, predict_labels(senses, theta_temp, train_bow)))

      test_results_prev = test_results
      predicted_labels_prev = predicted_labels
      predicted_labels = predict_labels(senses, theta_temp, test_bow)
      test_results = eval(test_labels, predicted_labels)

      #print "Result on test set: " + str(test_results)

      # Stopping condition when previous results exceed current
      # Rolls back to previous results and halts in such a case
      if (test_results_prev[0] > test_results[0]):
        #print "Previous test result of " + str(test_results_prev) + " exceeded current; rolling back to previous and stopping."
        #print "Final test accuracy: " + str(test_results_prev)
        write_predictions(predicted_labels_prev, "q4p4_pn.txt")
        return test_results_prev

      shuffle(indices)
    index = indices[counter % len(indices)]

    # Obtain predicted from argmax of scores for each class
    scoring = defaultdict(int)
    for sense in senses:
      for word in train_bow[index]:
        scoring[sense] += train_bow[index][word] * theta[sense][word]
    v = list(scoring.values())
    k = list(scoring.keys())
    yhat = k[v.index(max(v))]
   
    # If prediction is wrong, update weight vector
    correct_label = train_labels[index]
    if (yhat != correct_label):
      for word in train_bow[index]:
        # Updates for scores of predicted class
        m[yhat][word] += theta[yhat][word] * (counter - m_last_updated[yhat][word])
        m_last_updated[yhat][word] = counter
        theta[yhat][word] -= train_bow[index][word]
        m[yhat][word] -= train_bow[index][word]

        # Updates for scores of actual class
        m[correct_label][word] += theta[correct_label][word] * (counter - m_last_updated[correct_label][word])
        m_last_updated[correct_label][word] = counter
        theta[correct_label][word] += train_bow[index][word]
        m[correct_label][word] += train_bow[index][word]

    counter += 1


if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    ##running the classifier
    #training_scores = get_bow_naivebayes_training_scores(train_texts, train_labels, dev_texts, dev_labels)
    #test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
		#		dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    #print ("Naive Bayes training scores:\t" + str(training_scores))
    #print ("Naive Bayes test scores:\t" + str(test_scores) + "\n")
    #print ("Extended(1,0) Naive Bayes test scores:\t" + str(test_scores) + "\n")
    
    test_scores = run_bow_perceptron_classifier(train_texts, train_targets, train_labels,
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    print test_scores
