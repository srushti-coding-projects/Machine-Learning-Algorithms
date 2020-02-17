##########################################################################

# Copyright (c) 2018 Srushti Kokare
# sck@pdx.edu

##########################################################################

#Naive Bayes classifier for spam detection


import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def main():

    # Split spambase data into training and test data
    data = np.loadtxt('spambase.txt', delimiter=',', dtype=np.float32)
    training_spam = data[0:920]
    test_spam = np.concatenate((data[921:1813], data[0:28]), axis=0)
    training_notspam = data[1814:3194]
    test_notpsam = data[3195:4575]
    test_data = np.concatenate((test_spam, test_notpsam), axis=0)
    test_targets = test_data[:, 57]
    test_data = np.delete(test_data, 57, 1)

    # calculate prior class probabilities
    prob_spam = len(training_spam)/(len(training_spam) + len(training_notspam))
    prob_notspam = len(training_notspam) / (len(training_spam) + len(training_notspam))
    print('prior class probabilities:', prob_spam, ':', prob_notspam)

    # calculate mean and standard deviations for all features
    mean_spam = np.mean(training_spam[:, 0:57], axis=0)
    mean_notspam = np.mean(training_notspam[:, 0:57], axis=0)
    std_spam = np.std(training_spam[:, 0:57], axis=0)
    np.putmask(std_spam, std_spam < 0.0001, 0.0001)
    std_notspam = np.std(training_notspam[:, 0:57], axis=0)
    np.putmask(std_notspam, std_notspam < 0.0001, 0.0001)

    predictions = []
    for input in test_data:
        # calculate likelihood probabilities for spam class
        pr_spam = calculate_probabilities(input, mean_spam, std_spam)
        # calculate likelihood probabilities for not spam class
        pr_nspam = calculate_probabilities(input, mean_notspam, std_notspam)
        spam_class = classify(np.asarray(pr_spam), prob_spam)
        notspam_class = classify(np.asarray(pr_nspam), prob_notspam)
        if spam_class >= notspam_class:
            predictions.append(1)
        else:
            predictions.append(0)

    print('Accuracy:', accuracy_score(test_targets, predictions))
    print('Precision:', precision_score(test_targets, predictions))
    print('Recall:', recall_score(test_targets, predictions))
    print('CONFUSION MATRIX:')
    print(confusion_matrix(test_targets,predictions))


def classify(likelihood_prob, class_prob):
    '''
    :param likelihood_prob: likelihood probabilities for features given a class
    :param class_prob: prior probability of class
    :return:
    '''
    likelihood_prob[likelihood_prob == 0.0] = 0.00000000000000000000000000000000000000001
    likelihood_prob = np.sum(np.log(likelihood_prob))
    return likelihood_prob + class_prob


def calculate_probabilities(inputs, means, stds):
    '''
    :param inputs: features of a single input from test data
    :param means: means of features
    :param stds: standard deviation of features
    :return: likelihood probabilities for a class
    '''
    likelihood = []
    for input, mean, std in zip(inputs, means, stds):
        exp_part = -(((input - mean) * (input - mean))/(2 * std * std))
        mult_part = 1/(math.sqrt(2 * 3.142) * std)
        answer = mult_part * math.exp(exp_part)
        likelihood.append(answer)
    return likelihood


main()
