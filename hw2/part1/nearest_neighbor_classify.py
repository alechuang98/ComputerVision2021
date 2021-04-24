from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    K = 5

    dis = -2 * train_image_feats @ test_image_feats.T + np.sum(test_image_feats ** 2, axis=1) + np.sum(train_image_feats ** 2, axis=1)[:, np.newaxis]
    dis[dis < 0] = 0
    indx = np.argsort(dis, axis=0)

    test_predicts = []
    for j in range(indx.shape[1]):
        tmp = []
        for i in range(indx.shape[0]):
            tmp.append(train_labels[indx[i][j]])
        test_predicts.append(max(tmp, key=tmp.count))

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
