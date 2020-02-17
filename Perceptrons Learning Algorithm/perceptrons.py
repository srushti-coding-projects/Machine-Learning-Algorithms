##########################################################################

# Copyright (c) 2018 Srushti Kokare
# sck@pdx.edu

##########################################################################

# Perceptron Learning algorithm to classify handwritten digits on MNIST dataset

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Initialization of weight matrix

weight = np.empty((785, 10))


def main():

    # Read raw data from dataset
    imageset, targets = read_data_from_csv('mnist_train.csv')
    imageset_test, targets_test = read_data_from_csv('mnist_test.csv')

    # Train for 70 epochs with different learning rates

    for eta in [0.001, 0.01, 0.1]:
        global weight
        weight = np.random.random((785, 10))-0.5
        accuracy_train = []
        accuracy_test = []
        for epoch in range(70):
            for image, actual_target in zip(imageset, targets):

                # pass image data with targets to feed forward network
                predicted_target, result = forward_propagation(image)

                # Check for same targets and update weights
                if actual_target != predicted_target:
                    update_weights(image, result, actual_target, eta)

            # calculate accuracy of training dataset and test dataset

            predicted_list_train = get_predicted_list(imageset, targets)
            accuracy_train.append(accuracy_score(targets, predicted_list_train))

            predicted_list_test = get_predicted_list(imageset_test, targets_test)
            accuracy_test.append((accuracy_score(targets_test, predicted_list_test)))
            print(epoch, ':', accuracy_train[-1], accuracy_test[-1])

        # Calculation of confusion matrix

        print(confusion_matrix(targets, predicted_list_train))
        print(accuracy_train[-1])
        print(accuracy_test[-1])

        # Plot of training and test accuracies
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Testing and Test Accuracy over epochs')
        plt.ylim((0, 1))
        plt.plot(list(range(70)), accuracy_train, color='r', label='Accuracy for Training Set')
        plt.plot(list(range(70)), accuracy_test, color='b', label='Accuracy for Test Set')
        plt.savefig('output.png')
        plt.show()


# function to return the predicted list of targets
def get_predicted_list(imageset, targets):
    """

    :param imageset: image information
    :param targets: label of the image
    :return: list of the predicted targets
    """
    predicted_list = []
    for image, actual_target in zip(imageset, targets):
        predicted_target, result = forward_propagation(image)
        predicted_list.append(predicted_target)
    return predicted_list


# Read raw dataset from file and separate data and targets
def read_data_from_csv(filename):
    """

    :param filename: name of the file to read data
    :return: information of image and its labels
    """
    # Initialize small random weights as a matrix of [10,784]
    data = np.loadtxt(filename, delimiter=",")
    targets = data[:, 0]
    # insert the bias input
    imageset = np.insert(data[:, 1:]/255, 0, 1, axis=1)
    return imageset, targets


def forward_propagation(image):
    """

    :param image: inputs of image
    :return: predicted label and changed weights
    """
    image = image.reshape(1, 785)
    result = np.matmul(image, weight)
    predicated_target = np.argmax(result)
    return predicated_target, result


def update_weights(image, result, target_value, eta):
    """

    :param image: inputs of image
    :param result: previous weighted inputs
    :param target_value: target label
    :param eta: learning rate
    :return: none
    """
    target_matrix = np.zeros(10)
    target_matrix[int(target_value)] = 1
    output_matrix = np.zeros(10)
    indices = np.argwhere(result.flatten() > 0)
    output_matrix[indices.flatten()] = 1
    target_minus_output = np.subtract(target_matrix, output_matrix)
    a = np.multiply(eta, target_minus_output)
    a = a.reshape(10, 1)
    eta_input_matrix = np.dot(a, image.reshape(1, 785))
    eta_input_matrix = np.transpose(eta_input_matrix)
    global weight
    weight = np.add(weight, eta_input_matrix)


main()
