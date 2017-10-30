#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:21:55 2017

@author: vedparanjape
"""

import numpy as np
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
import random
import math

maxi = 999999999999.99


def create_clusters(distances, data, n, clusters_old):
    print ("n in clusters:"), n
    cluster = {}
    dist_local = distances.copy()
    # print ("data in create_clusters:"),data
    local_data = np.copy(data)
    for i in range(n):
        cluster[i] = None
    deleted = 0
    for row in range(n):
        if len(local_data) == 1:
            if (cluster[row] == None):
                cluster[row] = [clusters_old[deleted]]
            else:
                cluster[row].append(clusters_old[deleted])
        else:
            i = 0
            print ("row:"), row
            min_dist_from = np.argmin(distances[i])
            print ("min_dist_from:"), min_dist_from
            if (cluster[row] == None):
                cluster[row] = [clusters_old[deleted]]
            else:
                cluster[row].append(clusters_old[deleted])
            cluster[row].append(clusters_old[min_dist_from + deleted])

            distances = distances[i + 1:, i + 1:]  # np.concatenate((distances[:row],distances[row+1:]),axis = 0)
            local_data = local_data[i + 1:]  # np.concatenate((local_data[:row],local_data[row+1:]),axis = 0)
            deleted = deleted + 1
            distances = np.concatenate((distances[:min_dist_from - 1], distances[min_dist_from:]), axis=0)
            local_data = np.concatenate((local_data[:min_dist_from - 1], local_data[min_dist_from:]), axis=0)
            if min_dist_from < deleted:
                deleted = deleted + 1
            distances = np.delete(distances, min_dist_from - 1, 1)
            print ("local:"), local_data
            # local_data = np.delete(local_data,min_dist_from,1)
            # print cluster
            # print len(local_data)
            # i = i + 1
            # print distances.shape

    # print cluster
    # print ("\ndistances :"),len(distances)
    return (cluster, n)


def cal_centroids(clusters, n, columns):
    data_curr = []
    centroids_new = []
    dist_curr = np.zeros((n, n))
    for i in range(n):
        arr = np.asarray(clusters[i])
        data_curr.append(arr)
    for i in data_curr:
        print np.mean(i, axis=0)
        centroids_new.append(np.mean(i, axis=0))
    for i in range(n):
        for j in range(n):
            dist_curr[i][j] = distance.euclidean(centroids_new[i], centroids_new[j])
    # print ("\ncentroids_new:"),centroids_new
    return (centroids_new, dist_curr)


def cluster(data, rows, columns):
    # print data
    n = rows
    clusters = {}
    point_in_clus = {}
    for i in range(rows):
        clusters[i] = data[i]
        point_in_clus[i] = i
    # print clusters
    dist = np.zeros((rows, rows))
    # dist_matrix.reshape(rows,columns)
    for i in range(rows):
        for j in range(rows):
            if (i == j):
                dist[i][j] = maxi
            else:
                dist[i][j] = distance.euclidean(data[i], data[j])

    # print dist
    while n != 1:
        if n % 2 != 0:
            n = int(math.ceil(n / 2)) + 1
        else:
            n = int(math.ceil(n / 2))
        clusters, n = create_clusters(dist, data, n, clusters)
        data, dist = cal_centroids(clusters, n, columns)
        print ("n:"), n
        print ("clusters:"), clusters
        print ("\ndata:"), data
        # print point_in_clus


def main():
    iris = datasets.load_iris()
    X = iris.data
    feature_names = iris.feature_names
    y = iris.target
    target_names = iris.target_names

    # print X
    # X = np.delete(X,1,1)
    # print X
    # X1 = X[:,2:]
    # print X1
    # rows,columns = X1.shape
    # X1 = np.concatenate((X1[:0],X1[1:]),axis = 0)
    # print X1
    Z = np.array([[1, 2], [3, 10], [6, 6], [3, 7], [9, 10], [2, 7], [8, 1], [9, 3]], dtype=int)
    rows, columns = Z.shape
    cluster(Z, rows, columns)


if __name__ == "__main__":
    main()