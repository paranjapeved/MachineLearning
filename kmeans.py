import numpy as np
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
import random
from scipy.spatial import distance

def wcss(X,m):
    wcss_arr = np.zeros(len(X),dtype=float)
    for i in range(len(wcss_arr)):
        temp = 0
        for j in X[i]:
            temp = temp + distance.euclidean(j,m[i]) * distance.euclidean(j,m[i])
        wcss_arr[i] = temp
    return np.min(wcss_arr)

def assign_clusters(data, centroids_new, k):
    # dictionary for storing cluster allocation
    centroids_new = np.asarray(centroids_new)
    clusters = {}
    for i in range(k):
        clusters[i] = None
    # for i in range(k):
    #    clusters[i] = None

    for x in data:
        temp = np.zeros(k, dtype=float)
        for i in range(k):
            temp[i] = (np.linalg.norm(x - centroids_new[i, :]))
        cl = temp.argmin()
        if (clusters[cl] == None):
            clusters[cl] = [x]
        else:
            clusters[cl].append(x)
    return clusters


def converged(centroids_new, centroids_old):
    print "here"
    return (set([tuple(a) for a in centroids_new]) == set([tuple(a) for a in centroids_old]))


def cal_centroids(centers, clusters, k, c_cols):
    centers_new = np.zeros_like(centers)
    print centers_new
    for i in range(k):
        arr = np.asarray(clusters[i])
        if clusters[i] is not None:
            for j in range(c_cols):
                centers_new[i][j] = np.average(arr[:, j])
                # print np.average(arr[:,j])
    # print ("centers"),centers_new
    return centers_new


def euclidean_dist(a, b):
    leng = len(a)
    dist = 0.0
    for i in range(leng):
        dist = dist + (a[i] - b[i]) ** 2
    dist = np.sqrt(dist)
    return dist


def cluster(data, k):
    k = int(k)
    data = np.asarray(data)
    rows, cols = data.shape
    # initialize centroid to first k data points
    centroids_new = []
    for i in range(2, k + 2):
        centroids_new.append(data[i, :])

    centroids_new = np.asarray(centroids_new)
    # centroids_old = np.asarray(centroids_new)
    # print ("centroids:\n"),centroids_new
    # centroids_new = np.asarray(random.sample(data,k))
    centroids_old = np.asarray(random.sample(data, k))
    # print ("old centroids:"),centroids_

    c_rows, c_cols = centroids_new.shape
    clusters = {}
    print ("clusters:"), clusters
    n = 0
    while not converged(centroids_new, centroids_old):
        print ("iteration number:"), n
        centroids_old = centroids_new
        # assign data points to nearest clusters
        clusters = assign_clusters(data, centroids_new, k)
        # evaulate new centroids
        centroids_new = cal_centroids(centroids_old, clusters, k, c_cols)
        n = n + 1
        # print ("old centroids:"),centroids_old
        # print ("new centroids:"),centroids_new
    return (centroids_new, clusters)


iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names

X1 = X[:, 2:]
rows, columns = X1.shape
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(X1[:,0],X1[:,1],color = 'blue')
k = 3
# k = raw_input("Enter number of clusters")
wcss_arr = []
num = 1
while num < 5:
    final_centers, clusters = cluster(X1, num)
    print clusters
    gr = []
    for i in range(num):
        arr = np.asarray(clusters[i])
        gr.append(arr)
    wcss_arr.append(wcss(gr,final_centers))
    print wcss_arr
    num = num + 1

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("wcss")
ax.scatter(range(1,num),wcss_arr, color='blue')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(gr[0][:, 0], gr[0][:, 1], color='blue')
ax.scatter(gr[1][:, 0], gr[1][:, 1], color='red')
ax.scatter(gr[2][:, 0], gr[2][:, 1], color='green')
plt.show()









