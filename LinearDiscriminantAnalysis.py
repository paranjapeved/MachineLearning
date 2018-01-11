import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def LDA(X,II_0,II_1,nclass):
    X_dict = {}
    X_dict[0] = X[II_0]
    X_dict[1] = X[II_1]

    n = 19
    samples = 20

    #calculate mean vectors
    mean_vectors = {}
    for i in range(nclass):
        mean_vectors[i] = np.mean(X_dict[i],axis=0)



    #calculating scatters matrices

    #calculating within class scatter matrix
    S_within = np.zeros((n,n))
    for i in range(nclass):
        diff_sum = np.zeros((n,n))
        for j in range(samples):
            diff = (X_dict[i][j] - mean_vectors[i])
            diff = diff.reshape(n,1)
            diff_sum = diff_sum + diff.dot(diff.T)
        S_within = S_within + diff_sum

    #calculating between class scatter
    mean_overall = np.mean(X,axis=0)

    S_between = np.zeros((n,n))
    mean_overall = mean_overall.reshape((n,1))
    for i in range(nclass):
        mean_vectors[i] = mean_vectors[i].reshape((n,1))    #reshape each mean vector for each class
        #calculate S_between
        difference = mean_vectors[i] - mean_overall
        S_between = S_between + ((difference).dot(difference.T)*samples)

    #calculating the eigen values and eigen vectors for S_withine-1 x S_between
    S_within_inv = np.linalg.inv(S_within)
    eigen_vals,eigen_vectors = np.linalg.eig(S_within_inv.dot(S_between))



    #Finding the corresponding eigen vectors for largest eigen values
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))]

    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)


    eigen_vals = []
    for i in eigen_pairs:
        eigen_vals.append(i[0])



    print ("Variance described by first eigen vector in %:"),100*eigen_vals[0]/np.sum(eigen_vals)

    #Since first eigen vector can explain almost all variance without much loss of information, we take only first one

    W = eigen_pairs[0][1]

    #Returning the new principle axis
    return W



def main():
    file_name = "SCLC_study_output_filtered_2.csv"
    data_in = pd.read_csv(file_name,index_col=0)
    X = data_in.as_matrix()

    y = np.concatenate((np.zeros(20), np.ones(20)))

    II_0 = np.where(y == 0)
    II_1 = np.where(y == 1)

    II_0 = II_0[0]
    II_1 = II_1[0]

    nclasses = 2

    W = LDA(X,II_0,II_1,nclasses)
    print W

    #projecting original inputs to new priciple axis
    project = X.dot(W)
    project = project.real
    project = -project

    #scaling down the results to sklearn outputs
    project_scaled = (project/38255.1820948)-6
    #print project

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Results from applying LDA to cell line data(original)')
    ax.scatter(project[0:20], np.zeros(20), color='blue',label='NSCLC')
    ax.scatter(project[20:40], np.zeros(20), color='red',label='SCLC')
    ax.legend()
    fig.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Results from applying LDA to cell line data(down-scaled)')
    ax.scatter(project_scaled[0:20], np.zeros(20), color='blue', label='NSCLC')
    ax.scatter(project_scaled[20:40], np.zeros(20), color='red', label='SCLC')
    ax.legend()
    fig.show()

    #Using sklearn LDA
    sklearn_LDA = LinearDiscriminantAnalysis()
    sklearn_LDA_projection = sklearn_LDA.fit_transform(X,y)
    sklearn_LDA_projection = -sklearn_LDA_projection

    # plot the projections
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Results from applying sklearn LDA to cell line data')
    ax.scatter(sklearn_LDA_projection[0:20],np.zeros(20), color='blue', label='NSCLC')
    ax.scatter(sklearn_LDA_projection[20:40],np.zeros(20), color='red', label='SCLC')
    ax.legend()

    fig.show()



if __name__ == '__main__':
    main()