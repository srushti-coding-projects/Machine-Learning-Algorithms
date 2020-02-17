##########################################################################

# Copyright (c) 2018 Srushti Kokare
# sck@pdx.edu

##########################################################################

# Two Layer Perceptron to perform handwritten digit recognition
# Use of back-propagation with stochastic gradient descent to train the
# network.


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

hidden_output_wts = np.empty(0)
previous_output_wts = np.zeros(0)
input_hidden_wts = np.empty(0)
previous_hidden_wts = np.zeros(0)
counter = 1


def main():

    # Read training data set from the mnist_train.csv file and get images and their targets
    imageset, actual_targets = read_data_from_csv('mnist_train.csv')
    # Read test data set from the mnist_test.csv file and get images and their targets
    imageset_test, targets_test = read_data_from_csv('mnist_test.csv')

    # Train the model with changing the number of hidden units

    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 20, 0.1, 0.9)
    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 50, 0.1, 0.9)
    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 100, 0.1, 0.9)

    # Train the model with different momentum

    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 100, 0.1, 0)
    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 100, 0.1, 0.25)
    train_back_propagation(imageset, actual_targets, imageset_test, targets_test, 100, 0.1, 0.5)

    # Train the model with quarter and then training set of dataset
    # The training dataset is roughly balanced

    train_back_propagation(imageset[0: 15000, :], actual_targets[0:15000], imageset_test[0:15000, :],
                           targets_test[0:15000], 100, 0.1, 0.9)
    train_back_propagation(imageset[0: 30000, :], actual_targets[0:30000], imageset_test[0:30000, :],
                           targets_test[0:30000], 100, 0.1, 0.9)


def train_back_propagation(imageset, actual_targets, imageset_test, targets_test, hidden_units, eta, momentum):
    """

    :param imageset: input images in the training data set
    :param actual_targets: targets of the input images in the training data set
    :param imageset_test: input images in the test data set
    :param targets_test: targets of the input images in the test data set
    :param hidden_units: number of hidden units
    :param eta: learning rate
    :param momentum: momentum

    """

    # Initialise the weight matrices
    global input_hidden_wts, hidden_output_wts, previous_output_wts, previous_hidden_wts
    input_hidden_wts = np.random.random((imageset.shape[1], hidden_units)) - 0.5
    hidden_output_wts = np.random.random((hidden_units + 1, 10)) - 0.5
    previous_output_wts = np.zeros((hidden_units+1, 10))
    previous_hidden_wts = np.zeros((imageset.shape[1], hidden_units))
    accuracy_train = []
    accuracy_test = []
    training_predicted_list = []

    for epoch in range(50):
        for image, actual_target in zip(imageset, actual_targets):
            image = image.reshape(1, 785)

            # for every image input in the imageset, forward propogate the image at two different layers
            result_hidden_output, result_without_bias, result_input_hidden, probability_outputs,\
                predicted_target = propagate_two_layers(image)

            # If the target does not match the predicted target then calculate error at the output layer
            # and the hidden layer and update the weights.
            if actual_target != predicted_target:
                delta_output, delta_hidden_layer = calculate_error(result_hidden_output,
                                                                   actual_target, result_without_bias)
                update_output_weights(eta, delta_output, result_input_hidden, momentum)
                update_hidden_weights(eta, delta_hidden_layer, image, momentum)

        # Get the predicted output list for the training data set once the leraning has completed.
        # Calculate accuracy with the actual targets and append it to the accuracy_train list for 50 epochs

        training_predicted_list = get_predicted_list(imageset)
        accuracy_train.append(accuracy_score(actual_targets, training_predicted_list))

        # Forward propogate the test data set and get the predicted output list for the test data set once the
        # learning has completed.
        # Calculate accuracy with the actual targets and append it to the accuracy_test list for 50 epochs
        test_predicted_list = get_predicted_list(imageset_test)
        accuracy_test.append((accuracy_score(targets_test, test_predicted_list)))

    # Print the confusion matrix, final accuracy of training and test data
    print(confusion_matrix(actual_targets, training_predicted_list))

    # Print tha final accuracy of the training data set
    print(accuracy_train[-1])

    # Print tha final accuracy of the test data set
    print(accuracy_test[-1])

    # Plot the graph of training versus test data over 50 epochs
    plot_graph(accuracy_train, accuracy_test)


# Plot training and test accuracy
def plot_graph(accuracy_train, accuracy_test):
    """
    :param accuracy_train: List containing accuracy of training data set for all 50 epochs
    :param accuracy_test:  List containing accuracy of test data set for all 50 epochs

    Plots a graph of accuracy of training data set and test data set over number of epochs
    """
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over epochs')
    plt.ylim((0, 1))
    plt.plot(list(range(50)), accuracy_train, color='r', label='Accuracy for Training Set')
    plt.plot(list(range(50)), accuracy_test, color='b', label='Accuracy for Test Set')
    global counter
    plt.savefig("output"+str(counter)+".png")
    counter = counter + 1
    plt.show()


# This method returns the predicted target value after feeding input forward through two layer
def get_predicted_list(imageset):
    """

    :param imageset: input of images in training or test data
    :return: list of output predictions

    Input Image data set id fed forward after the model is trained to get the predicted output list
    """
    predicted_list = []
    for image in imageset:
        image = image.reshape(1, 785)
        result_hidden_output, result_without_bias, result_input_hidden, \
            probability_outputs, predicted_target = propagate_two_layers(image)
        predicted_list.append(predicted_target)
    return predicted_list


def propagate_two_layers(image):
    """

    :param image: Single input image in the entire imageset of training or test data.
    :returns: result_hidden_output : output of hidden layer to output unit
    result_without_bias : result of input to hidden layer without bias
    result_input_hidden : result of input to hidden layer
    probability_outputs : results at output layer converted into probability outputs
    predicted_target : predicted target for the given input image


    """
    global input_hidden_wts, hidden_output_wts
    result_without_bias = forward_propagate(image, input_hidden_wts)
    result_input_hidden = np.insert(result_without_bias[0:], 0, 1, axis=1)
    result_hidden_output = forward_propagate(result_input_hidden, hidden_output_wts)
    probability_outputs, predicted_target = softmax(result_hidden_output)
    return result_hidden_output, result_without_bias, result_input_hidden, probability_outputs, predicted_target


def update_hidden_weights(eta, delta_hidden_layer, image, momentum):
    """

    :param eta: learning rate
    :param delta_hidden_layer: error at the hidden layer
    :param image: input image
    :param momentum: momentum

    Updates the weight matrix between input and hidden layer
    """
    global input_hidden_wts, previous_hidden_wts
    eta_delta_input = np.matmul(np.multiply(eta, delta_hidden_layer), image)
    input_hidden_wts = np.add(input_hidden_wts, np.add(eta_delta_input.transpose(),
                                                       np.multiply(momentum, previous_hidden_wts)))
    previous_hidden_wts = [0]
    previous_hidden_wts = np.add(eta_delta_input.transpose(), np.multiply(momentum, previous_hidden_wts))


def update_output_weights(eta, delta_output, result_input_hidden, momentum):
    """

    :param eta: learning rate
    :param delta_output: erro at output layer
    :param result_input_hidden: result obtained at the input to hidden layer
    :param momentum: momentum

    Updates the weight matrix between hidden and output layer
    """
    global hidden_output_wts, previous_output_wts
    eta_delta_input = np.matmul(np.multiply(eta, delta_output), result_input_hidden)
    hidden_output_wts = np.add(hidden_output_wts, np.add(eta_delta_input.transpose(),
                                                         np.multiply(momentum, previous_output_wts)))
    previous_output_wts = [0]
    previous_output_wts = np.add(eta_delta_input.transpose(), np.multiply(momentum, previous_output_wts))


def calculate_error(result_hidden_output, actual_target, result_without_bias):
    """

    :param result_hidden_output: result at the hidden to output layer
    :param actual_target: actual target of that input image
    :param result_without_bias: result at the input to hidden layer without the bias
    :return: delta_output: error calculated at the output layer
            delta_hidden_layer: error calculated at the hidden layer
    """
    target = np.full(result_hidden_output.shape, 0.1)
    target = target.flatten()
    np.put(target, [actual_target], [0.9])
    target = target.reshape(1, 10)
    delta_output = np.multiply(np.multiply(result_hidden_output, 1 - result_hidden_output),
                               target - result_hidden_output)
    delta_output = delta_output.reshape(10, 1)
    weight_kj = np.delete(hidden_output_wts, 0, axis=0)
    weightkj_delta = np.matmul(weight_kj, delta_output)
    result = np.multiply(result_without_bias, 1 - result_without_bias)
    delta_hidden_layer = np.multiply(result.transpose(), weightkj_delta)
    return delta_output, delta_hidden_layer


def forward_propagate(image, weight):
    """

    :param image: inputs with a bias
    :param weight: weight matrix
    :return: returns sigmoid result of the forward propogation
    """
    result = np.matmul(image, weight)
    sigmoid_result = sigmoid(result)
    return sigmoid_result


def softmax(result_hidden_output):
    """

    :param result_hidden_output: result at the output layer
    :return: probability_outputs: probability ouptuts
             predicted_output: predicted output

    Converts the outputs into probabilities
    """
    output_exp = np.exp(result_hidden_output)
    probability_outputs = output_exp / np.sum(output_exp)
    predicted_output = np.argmax(probability_outputs)
    return probability_outputs, predicted_output


# Read dataset from csv file and separate image features and targets
def read_data_from_csv(filename):
    """

    :param filename: name of file to read
    :return: imageset: all the input images in the files
            actual_targets: the targets of the input images
    """
    data = np.loadtxt(filename, delimiter=",")
    actual_targets = data[:, 0]
    imageset = np.insert(data[:, 1:]/255, 0, 1, axis=1)
    return imageset, actual_targets


main()
