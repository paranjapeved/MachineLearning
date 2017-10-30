#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 08:44:44 2017

@author: vedparanjape
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import csv
from numpy import genfromtxt
from matplotlib.mlab import PCA


def pca(X1):

    print ("X:"), X1
    # calculating mean
    means = X1.mean(axis=0)
    # mean centering
    X = X1 - means

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0],X[:,1])
    plt.show()

    #calculating the covariance of data
    cov_x = np.cov(X,rowvar=False)

    #calculating eigen values and corresponding vectors
    eig_val,eig_vec = np.linalg.eig(cov_x)

    eigvec_t = eig_vec.T
    II = eig_val.argsort()[::-1]
    eig_val = eig_val[II]
    eig_vec = eig_vec[:, II]

    #plotting the scree plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('scree plot')
    ax.scatter(range(len(eig_val)), eig_val, color='blue')
    fig.show()

    #plotting the loadings plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('loadings plot')
    ax.scatter(eig_vec[:, 0], eig_vec[:, 1], color='blue')
    ax.set_xlabel('PC1')
    #ax.set_ylabel('PC2')
    fig.show()

    #X1 = genfromtxt('dataset_1.csv',dtype=float,skip_header = 1,delimiter=',')
    Y = X.dot(eig_vec)
    #print ("covari:"),np.cov(Y[:,0],Y[:,1])

    #plotting the scores plot for PC1 and PC2
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_title('scores plot')
    ax.scatter(Y[:,0],Y[:,1])
    plt.show()

    pcaResult = {'data': X1,
                  'mean_centered_data': X,
                  'PC_variance': eig_val,
                  'loadings': eig_vec,
                  'scores': Y}
    return pcaResult

def main():
    #read data from csv
    X1 = genfromtxt('SCLC_study_output_filtered.csv', skip_header=1, delimiter=',')
    #delete headers
    X1 = np.delete(X1,0,1)
    print X1
    num_rows,num_cols = X1.shape
    X = X1

    results = pca(X)

    #regenerating original data from PCs, reduces noise in the data
    X_regenerated = results['scores'].dot(results['loadings'].T)
    print ("\nregenerated:"),X_regenerated[:,:2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("after regenation")
    ax.scatter(X_regenerated[:, 0], X_regenerated[:, 1])
    plt.show()


if __name__ == "__main__":
    main()


