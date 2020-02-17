##########################################################################

# Copyright (c) 2018 Srushti Kokare
# sck@pdx.edu

##########################################################################


# Implementation of Decision tree to predict vote for the party


import numpy as np
from sklearn.metrics import accuracy_score
import math


# Class for tree node
class Node(object):

    def __init__(self, feature_data, target_data, is_root, is_leaf, split_feature, runningListOfAttr, leaf_value):
        self.feature_data = feature_data
        self.target_data = target_data
        self.left_child = None
        self.right_child = None
        self.split_feature = split_feature
        self.runningListOfAttr = runningListOfAttr
        self.leaf = is_leaf
        self.leaf_value = leaf_value
        self.is_root = is_root


# Data preprocessing
# Convert the labels as well as features into numbers
def preprocess(raw_data):
    """

    :param raw_data:
    :return:

    n set to 0 : no
    y set to 1: yes
    ? set to -1: no Vote

    republican party set to 0
    democrat party set to 1
    """

    raw_data[raw_data == 'n'] = 0
    raw_data[raw_data == 'y'] = 1
    raw_data[raw_data == '?'] = -1

    raw_data[raw_data == 'republican'] = 0
    raw_data[raw_data == 'democrat'] = 1

    return raw_data


def count_samples(samples):
    """

    :param samples:
    :return:
    """
    unique, counts = np.unique(samples, return_counts=True)
    return dict(zip(unique, counts))


def calculate_entropy(class1, class2):
    """

    :param class1:
    :param class2:
    :return:
    """
    if class1 == 0 and class2 == 0:
        return 0
    prob_1 = class1 / (class1 + class2)
    prob_2 = class2 / (class1 + class2)
    if prob_1 <= 0:
        prob_1 = 0.0001
    if prob_2 <= 0:
        prob_2 = 0.0001
    entropy = - (prob_1 * math.log(prob_1, 2)) - (prob_2 * math.log(prob_2, 2))
    return entropy


def calculate_entropies_two_attributes(freq_table):
    '''

    :param freq_table:
    :return:
    '''
    total = freq_table[0, 0] + freq_table[0, 1] + freq_table[1, 0] + freq_table[1, 1] + freq_table[2, 0] + freq_table[2,1]
    prob_yes = (freq_table[0, 0] + freq_table[0, 1])/total
    prob_no = (freq_table[1, 0] + freq_table[1, 1])/total
    prob_novote = (freq_table[2, 0] + freq_table[2, 1])/total
    e1 = calculate_entropy(freq_table[0, 0], freq_table[0, 1])
    e2 = calculate_entropy(freq_table[1, 0], freq_table[1, 1])
    e3 = calculate_entropy(freq_table[2, 0], freq_table[2, 1])
    entropy_two_attribute = ((prob_yes * e1) + (prob_no * e2) + (prob_novote * e3))
    return entropy_two_attribute


def create_frequency_table(features1, targets1):
    '''

    :param features1:
    :param targets1:
    :return:
    '''
    rep_no = 0
    rep_yes = 0
    rep_novote = 0
    dem_no = 0
    dem_yes = 0
    dem_novote = 0
    for i in range(0, len(targets1)):
        if targets1[i] == 0:
            if features1[i] == 0:
                rep_no = rep_no+1
            if features1[i] == 1:
                rep_yes = rep_yes+1
            if features1[i] == -1:
                rep_novote = rep_novote+1
        if targets1[i] == 1:
            if features1[i] == 0:
                dem_no = dem_no+1
            if features1[i] == 1:
                dem_yes = dem_yes+1
            if features1[i] == -1:
                dem_novote = dem_novote+1
    freq_table = np.array([[rep_yes, dem_yes], [rep_no, dem_no], [rep_novote, dem_novote]])
    return freq_table


def get_split_point(targets1, features1):
    '''

    :param targets1:
    :param features1:
    :return:
    '''
    counted_samples = count_samples(targets1)
    entropy_one_attribue = calculate_entropy(counted_samples[0], counted_samples[1])

    # Calculate entropy using two attribues
    entropies_two_attribue = []

    features_index = features1.shape[1]
    for x in range(0, features_index):
        freq_table = create_frequency_table(features1[:, x], targets1)
        entropy = calculate_entropies_two_attributes(freq_table)
        entropies_two_attribue.append(entropy)

    information_gain = np.subtract(entropy_one_attribue, entropies_two_attribue)

    max_information_gain = np.argmax(information_gain)
    return max_information_gain


def split_data(targets1, features1, split_point, parentnode):
    '''

    :param targets1:
    :param features1:
    :param split_point:
    :param parentnode:
    :return:
    '''
    split_point_data = features1[:, split_point]
    features = np.delete(features1, [split_point], axis=1)
    parentnode.split_feature = parentnode.runningListOfAttr[split_point]

    features_left_child = []
    features_left_cnt = 0
    targets_left_child = []
    features_right_child = []
    features_right_cnt = 0
    targets_right_child = []

    for i in range(0, len(split_point_data)):
        if split_point_data[i] <= 0:
            features_left_child.append(features[i])
            targets_left_child.append(targets1[i])
            features_left_cnt = features_left_cnt + 1

        if split_point_data[i] >= 1:
            features_right_child.append(features[i])
            targets_right_child.append(targets1[i])
            features_right_cnt = features_right_cnt + 1

    child_attr_list = parentnode.runningListOfAttr[:]
    child_attr_list.pop(split_point)
    if features_left_cnt > 0:
        parentnode.left_child = Node(np.asarray(features_left_child), np.asarray(targets_left_child),
                                     0, 0, None, child_attr_list, 0)
    if features_right_cnt > 0:
        parentnode.right_child = Node(np.asarray(features_right_child), np.asarray(targets_right_child),
                                      0, 0, None, child_attr_list, 0)
    parentnode.leaf = 0


def build_tree(parentnode):
    '''

    :param parentnode:
    :return:
    '''
    if np.all(parentnode.target_data == 0) or np.all(parentnode.target_data == 1):
        if np.all(parentnode.target_data) is False:
            parentnode.left_child = Node(None, None, 0, 1, None, None, 'R')

        if np.all(parentnode.target_data) is True:
            parentnode.right_child = Node(None, None, 0, 1, None, None, 'D')

        return parentnode
    elif parentnode.feature_data.shape[1] == 0:
        parentnode.right_child = Node(None, None, 0, 1, None, None, 'D')
        return parentnode
    else:
        split_point = get_split_point(parentnode.target_data, parentnode.feature_data)

        split_data(parentnode.target_data, parentnode.feature_data, split_point, parentnode)

        if parentnode.left_child is not None and parentnode.is_root == 0:
            build_tree(parentnode.left_child)
        if parentnode.right_child is not None and parentnode.is_root == 0:
            build_tree(parentnode.right_child)
        return parentnode


def print_tree(Tree):
    '''

    :param Tree:
    :return:
    '''
    if Tree.leaf == 0:
        print("Parent Node Split Feature = ", Tree.split_feature)
        if Tree.left_child is not None:
            print("<=.Left children")
            print_tree(Tree.left_child)
        if Tree.right_child is not None:
            print(">=Right children")
            print_tree(Tree.right_child)
    elif Tree.leaf:
        print("THis is the leaf node")
        print(Tree.leaf_value)
        return
    else:
        print("Error Condition")
        input()
    return


def make_predictions(test_data_features, feature_attr_name, final_tree, label_num):

    prediction_arr = []
    for i in range(0, len(test_data_features)):
        get_prediction(test_data_features[i], feature_attr_name, final_tree, prediction_arr, label_num)
    return prediction_arr


def get_prediction(test_data_feature, feature_attr_name, tree,prediction_arr, label_num):
    '''

    :param test_data_feature:
    :param feature_attr_name:
    :param tree:
    :param prediction_arr:
    :param label_num:
    :return:
    '''
    if tree.leaf == 0:
        search_key = get_feature_num(tree.split_feature, feature_attr_name, label_num)
        if tree.split_feature is None:
            if tree.left_child is not None:
                get_prediction(test_data_feature, feature_attr_name, tree.left_child, prediction_arr, label_num)
            if tree.right_child is not None:
                get_prediction(test_data_feature, feature_attr_name, tree.right_child, prediction_arr, label_num)
        else:
            if test_data_feature[search_key] <= 0:
                get_prediction(test_data_feature, feature_attr_name, tree.left_child, prediction_arr, label_num)
            if test_data_feature[search_key] >= 1:
                get_prediction(test_data_feature, feature_attr_name, tree.right_child, prediction_arr, label_num)
    elif tree.leaf:
        prediction_arr.append(tree.leaf_value)
        return
    else:
        print("Error Condition in prediction")


def get_feature_num(search_attr_name, feature_attr_name, label_num):
    '''

    :param search_attr_name:
    :param feature_attr_name:
    :param label_num:
    :return:
    '''
    for i in range(0, len(feature_attr_name)):
        if feature_attr_name[i] == search_attr_name:
            if label_num < i:
                i = i-1
            return i


def calculate_accuracy(actual_targets, predicted_targets):
    '''

    :param actual_targets:
    :param predicted_targets:
    :return:
    '''
    for i in range(0, len(predicted_targets)):
        if predicted_targets[i] == 'D':
            predicted_targets[i] = 1
        if predicted_targets[i] == 'R':
            predicted_targets[i] = 0
    return accuracy_score(predicted_targets, actual_targets.tolist())


def prune_tree(Tree, prune_level, level, prune_done):
    '''

    :param Tree:
    :param prune_level:
    :param level:
    :param prune_done:
    :return:
    '''
    if Tree.leaf == 0:
        if level == prune_level-1 and prune_done == 0:
            Tree.left_child = None
            Tree.right_child = None
            Tree.leaf = 1
            Tree.leaf_value = 'R'
            prune_done = 1
            print("Pruning done")
            return prune_done
        else:
            if Tree.left_child is not None:
                level = level + 1
                prune_done = prune_tree(Tree.left_child, prune_level, level, prune_done)
    elif Tree.leaf:
        level = level + 1
        return prune_done
    else:
        print("Error Condition")
        input()
    return


def get_depth(Tree, tree_level, depth_done):
    '''

    :param Tree:
    :param tree_level:
    :param depth_done:
    :return:
    '''
    if Tree.leaf == 0:
        if Tree.left_child is None:
            return tree_level
        if Tree.left_child is not None:
            tree_level = tree_level + 1                 # increment level of the tree
            tree_level = get_depth(Tree.left_child, tree_level, depth_done)  # get depth of left subtree
    elif Tree.leaf:
        tree_level = tree_level + 1
        print("Max Depth of tree level is ", tree_level)
        return tree_level
    else:
        print("Error Condition")
        input()


def main():
    raw_data = np.genfromtxt('data.txt', delimiter=',', dtype='str')
    processed_data = preprocess(raw_data)           # clean and process the raw dataset
    attribute_names = processed_data[0, :]

    global dictOfAttributes
    # dictionary of names of attributes in the dataset
    dictOfAttributes = {i: attribute_names[i] for i in range(0, len(attribute_names))}

    vote_num_training_data = 348        # splitting of dataset. 10 percent data is for testing
                                        # and other 10 percent is for pruning
    vote_num_total_data = 435
    vote_label_num = 0
    vote_ListOfAttr = list(dictOfAttributes.values())
    del vote_ListOfAttr[vote_label_num]

    vote_training_data = processed_data[1:vote_num_training_data].astype(int)
    vote_test_data = processed_data[vote_num_training_data + 1:391].astype(int)
    vote_prune_data = processed_data[392:vote_num_total_data].astype(int)

    vote_test_data_labels = vote_test_data[:, 0]
    vote_test_data_features = vote_test_data[:, :]
    vote_test_data_features = np.delete(vote_test_data_features, [vote_label_num], axis=1)

    vote_prune_data_labels = vote_prune_data[:, 0]
    vote_prune_data_features = vote_prune_data[:, 1:]

    global vote_features,vote_targets
    vote_targets = vote_training_data[:, 0]
    vote_features = vote_training_data[:, :]
    vote_features = np.delete(vote_features, [vote_label_num], axis=1)

    vote_targets = vote_targets.reshape(vote_num_training_data-1,1)

    vote_rootnode = Node(vote_features, vote_targets, 0, 0, None, vote_ListOfAttr, 'D')

    build_tree(vote_rootnode)               # build tree 

    vote_level = 0
    depth_done = 0
    
    # get depth of the original tree to predict party
    vote_level = get_depth(vote_rootnode,vote_level,depth_done)
    vote_predictions_data = vote_training_data[:,:]
    vote_predictions_data = np.delete(vote_predictions_data, [vote_label_num], axis=1)

    vote_predicitions_test_data = make_predictions(vote_test_data_features,attribute_names, vote_rootnode,vote_label_num)
    vote_accuracy_test_data = calculate_accuracy(vote_test_data_labels, vote_predicitions_test_data)

    vote_predicitions_training_data = make_predictions(vote_predictions_data, attribute_names, vote_rootnode,vote_label_num)
    vote_accuracy_training_data = calculate_accuracy(vote_training_data[:, 0], vote_predicitions_training_data)

    vote_predicitions_prune_data = make_predictions(vote_prune_data_features, attribute_names, vote_rootnode,
                                                       vote_label_num)
    vote_accuracy_prune_data = calculate_accuracy(vote_prune_data_labels, vote_predicitions_prune_data)


    print("ACCURACIES")
    print("Accuracy over test data  with class labels as republic or democrat", vote_accuracy_test_data)
    print("Accuracy over training data with class labels as republic or democrat", vote_accuracy_training_data)
    print("Accuracy over pruning data with class labels as republic or democrat", vote_accuracy_prune_data)


    #-----------------------------------------------------------------------------------------------------

    party_num_training_data = 348
    party_num_total_data = 435
    party_label_num = 5
    party_ListOfAttr = list(dictOfAttributes.values())
    del party_ListOfAttr[party_label_num]

    party_training_data = processed_data[1:party_num_training_data].astype(int)
    party_test_data = processed_data[party_num_training_data+1:party_num_total_data].astype(int)

    global party_training_data_features,party_training_data_labels
    party_training_data_labels = party_training_data[:,party_label_num]
    np.place(party_training_data_labels, party_training_data_labels == -1, [0])

    party_training_data_features = np.delete(party_training_data,[party_label_num],axis=1)

    party_test_data_labels = party_test_data[:,party_label_num]
    np.place(party_test_data_labels, party_test_data_labels == -1, [0])
    party_test_data_features = np.delete(party_test_data, [party_label_num], axis=1)

    party_rootnode = Node(party_training_data_features, party_training_data_labels, 0, 0, None, party_ListOfAttr, 'D')

    build_tree(party_rootnode)


    print("===============================Height of the tree to predict vote on physician free freeze===========================")
    print("Getting depth of party tree")
    party_level = 0
    party_depth_done = 0
    party_level = get_depth(party_rootnode, party_level, party_depth_done)

    party_predicitions_test_data = make_predictions(party_test_data_features, attribute_names, party_rootnode,party_label_num)
    party_accuracy_test_data = calculate_accuracy(party_test_data_labels, party_predicitions_test_data)

    party_predicitions_training_data = make_predictions(party_training_data_features, attribute_names, party_rootnode,party_label_num)
    party_accuracy_training_data = calculate_accuracy(party_training_data_labels, party_predicitions_training_data)

    print("=========================================Accuracies====================================================================")
    print("Accuracy over test data to predict vote on physician fee freeze", party_accuracy_test_data)
    print("Accuracy over training data to predict vote on physician fee freeze", party_accuracy_training_data)

    #===================================================================================================================
    print("Now pruning tree to predict party")
    prune_level = 4
    level_start = 0
    prune_done_start = 0
    prune_done_end = prune_tree(vote_rootnode, prune_level, level_start, prune_done_start)
    print("==================Pruning tree to predict party======================================================================")
    print_tree(vote_rootnode)

    print("=======================Height of the pruned tree====================================================================")
    print("Getting depth of prune vote tree")
    prune_vote_level = 0
    prune_vote_depth_done = 0
    prune_vote_level = get_depth(vote_rootnode, prune_vote_level, prune_vote_depth_done)

    #====================================================================================================================
    prune_predicitions_test_data = make_predictions(vote_test_data_features, attribute_names, vote_rootnode,
                                                    party_label_num)
    prune_accuracy_test_data = calculate_accuracy(vote_test_data_labels, prune_predicitions_test_data)

    prune_predicitions_training_data = make_predictions(vote_features, attribute_names, vote_rootnode, party_label_num)
    prune_accuracy_training_data = calculate_accuracy(vote_training_data[:, 0], prune_predicitions_training_data)

    prune_predicitions_prune_data = make_predictions(vote_prune_data_features, attribute_names, vote_rootnode,
                                                     party_label_num)
    prune_accuracy_prune_data = calculate_accuracy(vote_prune_data_labels, prune_predicitions_prune_data)

    print("==========================Accuracy after pruning tree to predict party============================================")
    print("Accuracy over test data  on pruned tree for party", prune_accuracy_test_data)
    print("Accuracy over training data on pruned tree for party", prune_accuracy_training_data)
    print("Accuracy over pruning data on pruned tree for party", prune_accuracy_prune_data)

    #===============================================================================================================================
    
    #pruning tre to predict vote and party

    #===================================================================================================================
    print("Now pruning tree to predict party")
    prune_level = 4
    level_start = 0
    prune_done_start = 0
    prune_done_end = prune_tree(party_rootnode, prune_level, level_start, prune_done_start)
    print("===========================PRUNED TREE to predict physician fee")
    print_tree(party_rootnode)

    print("=========================Height of pruned tree=================================")
    print("Getting depth of prune party tree")
    prune_party_level = 0
    prune_party_depth_done = 0
    prune_party_level = get_depth(party_rootnode, prune_party_level, prune_party_depth_done)


main()
