"""
CMSC723 / INST725 / LING723 -- Fall 2016
Project 1: Implementing Word Sense Disambiguation Systems
"""

from collections import defaultdict
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
		print '>>>> invalid input !!! <<<<<'

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

"""
Trains a naive bayes model with bag of words features and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.
"""
def run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	
	"""
	**Your final classifier implementation of part 2 goes here**
	"""
	pass


# Creates bags of words for a set from the original lists of sentence words
def make_bow(texts):
  bow = []
  for text in texts:
    next_bow = defaultdict(int)
    for word in text:
      next_bow[word] += 1
    bow.append(next_bow)
  return bow

def get_predicted_labels(senses, theta, bow):
  labels = []
  for i in range(0, len(bow)):
    scoring = defaultdict(int)
    for sense in senses:
      for word in bow[i]:
        scoring[sense] += bow[i][word] * theta[sense][word]
    v = list(scoring.values())
    k = list(scoring.keys())
    label = k[v.index(max(v))]
    labels.append(label)
  return labels

def calculate_accuracy(senses, theta, bow, labels):
  correct = 0
  incorrect = 0
  for i in range(0, len(bow)):
    scoring = defaultdict(int)
    for sense in senses:
      for word in bow[i]:
        scoring[sense] += bow[i][word] * theta[sense][word]
    v = list(scoring.values())
    k = list(scoring.keys())
    label = k[v.index(max(v))]
    if (label == labels[i]):
      correct += 1
    else:
      incorrect += 1
  #print "#Correct: " + str(correct) + ", #Incorrect: " + str(incorrect)
  return float(correct) / (correct + incorrect)

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
    theta[sense] = defaultdict(int)
    m[sense] = defaultdict(int)
    m_last_updated[sense] = defaultdict(int)

  # Create training and test bags of words collection
  train_bow = make_bow(train_texts)
  test_bow = make_bow(test_texts)

  # Initialize list of indices used for randomizing training instance order
  indices = []
  for i in range(0, len(train_texts)):
    indices.append(i)

  # Main perceptron loop
  counter = 0
  while (True): # TODO: Later change to be while (accuracy not decreasing) or similar; NEEDS STOPPING CONDITION!!!!

    # Currently evaluates accuracy on training set after rolls through whole training set once
    if (counter % len(train_texts) == 0 and counter > 0):
      m_temp = dict()
      theta_temp = dict()
      for sense in senses:
        # Needs use of last_updated part too, doesn't it...
        m_temp[sense] = defaultdict(int)
        theta_temp[sense] = defaultdict(int)
        for word in m[sense]:
          m_temp[sense][word] = m[sense][word] + theta[sense][word] * (counter - m_last_updated[sense][word])
          theta_temp[sense][word] = m_temp[sense][word] / counter
      #print "Current accuracy on training set: " + str(calculate_accuracy(senses, theta_temp, train_bow, train_labels))
      #print "Current accuracy on test set: " + str(calculate_accuracy(senses, theta_temp, test_bow, test_labels))
      print "Result of eval on training set: " + str(eval(train_labels, get_predicted_labels(senses, theta_temp, train_bow)))
      print "Result of eval on test labels vs gold labels: " + str(eval(test_labels, get_predicted_labels(senses, theta_temp, test_bow)))
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

  # TODO: Need to use all value accumulations based on m_last_updated properly when break out of loop!!

  # Obtain final weights from running average
  for sense in senses:
    for word in m[sense]:
      theta[sense][word] = m[sense][word] / counter

  # Use trained theta on test set and return
  return calculate_accuracy(senses, theta, make_bow(test_texts), test_labels);


"""
Trains a naive bayes model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_naivebayes_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass

"""
Trains a perceptron model with bag of words features  + two additional features 
and computes the accuracy on the test set

train_texts, train_targets, train_labels are as described in read_dataset above
The same thing applies to the reset of the parameters.

"""
def run_extended_bow_perceptron_classifier(train_texts, train_targets,train_labels, 
				dev_texts, dev_targets,dev_labels, test_texts, test_targets, test_labels):
	"""
	**Your final implementation of Part 4 with perceptron classifier**
	"""
	pass


if __name__ == "__main__":
    # reading, tokenizing, and normalizing data
    train_labels, train_targets, train_texts = read_dataset('train')
    dev_labels, dev_targets, dev_texts = read_dataset('dev')
    test_labels, test_targets, test_texts = read_dataset('test')

    #running the classifier
    #test_scores = run_bow_naivebayes_classifier(train_texts, train_targets, train_labels, 
		#		dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)
    test_scores = run_bow_perceptron_classifier(train_texts, train_targets, train_labels, 
				dev_texts, dev_targets, dev_labels, test_texts, test_targets, test_labels)

    #print test_scores
