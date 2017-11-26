import pandas as pd
from sklearn import datasets
import numpy as np
import math
import matplotlib.pyplot as plt

def hypothesis(theta,x):
    Z = 0
    for i in range(len(theta)):
        Z = Z + (x[i] * theta[i])
    return (float(1) / (float(1) + math.e**(-Z)))

def cost_function(X,Y,theta):
    m = len(Y)
    #temp = np.zeros((m,1));
    error_sum = 0
    for i in range(m):
        xi = X[i]
        hx = hypothesis(theta,xi)
        error_sum = error_sum + (Y[i] * math.log(hx) + (1-Y[i]) * np.log(1-hx))
    J_theta = (-1/m) * error_sum
    return J_theta

def cost_differentiation(X,Y,theta,j,m,alpha):
    errors_sum = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hx = hypothesis(theta,xi)
        temp = (hx - Y[i])*xij
        errors_sum = errors_sum + temp
    return (alpha/m)*errors_sum

def gradientDescent(X,Y,theta_old,m,alpha):
    #theta_new = np.zeros((len(theta_old),1))
    theta_new = []
    for j in range(len(theta_old)):
        derivative = cost_differentiation(X,Y,theta_old,j,m,alpha)
        theta_val = theta_old[j] - (derivative)
        print ("theta val:"),theta_val
        theta_new.append(theta_val)
    return theta_new

def logistic_regression(X,Y,theta,alpha,iters):
    m = len(Y)
    theta_rec = []
    costfunction_values = []
    theta_rec.append(theta)
    #print cost_function(X,Y,old_theta)
    for i in range(iters):
        theta_new = gradientDescent(X,Y,theta,m,alpha)
        theta = theta_new
        theta_rec.append(theta)
        costfunction_values.append(cost_function(X,Y,theta))
        print "Iteration number:",i+1
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(costfunction_values,np.array([1]), '.', color='red', label='J(theta)')
    # ax.legend()
    print theta_rec
    print costfunction_values

def main():
    #read data into dataframe
    data_df = pd.read_csv("banking.csv",header=0)

    #data_df = datasets.load_diabetes()
    #X1 = data_df.data[:,2]
    categorical_data = data_df[['job', 'marital', 'default', 'housing', 'loan', 'poutcome', 'y']]

    # Create dummy variables
    categorical_dummy = pd.get_dummies(categorical_data,
                                             columns=['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
    categorical_dummy.head()

    # remove columns with unknown values
    all_columns = list(categorical_dummy)
    indices_for_unknown = [i for i, s in enumerate(all_columns) if 'unknown' in s]

    data_for_analysis_final = categorical_dummy.drop(categorical_dummy.columns[indices_for_unknown], axis=1)

    # get training and testing sets
    X = data_for_analysis_final.iloc[:, 1:]
    y = data_for_analysis_final.iloc[:, 0]

    num_of_samples = X.shape[0]
    num_of_test_samples = int(0.20 * num_of_samples)

    test_sample_index = np.int_(num_of_samples * np.random.rand(num_of_test_samples))
    train_sample_index = np.setdiff1d(np.arange(0, num_of_samples, 1), test_sample_index)
    X0 = np.ones((int(0.8 * num_of_samples), 1))

    X_train = X.iloc[train_sample_index, :]
    X_test = X.iloc[test_sample_index, :]
    Y_train = y.iloc[train_sample_index]
    y_test = y.iloc[test_sample_index]

    X_train["X0"] = [1] * (len(X_train))
    alpha = 0.01


    cols = X_train.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X_train = X_train[cols]
    X_train = X_train.as_matrix(columns=None)
    Y_train = Y_train.as_matrix(columns=None)
    rows,cols = X_train.shape
    init_theta = [1] * cols

    logistic_regression(X_train,Y_train,init_theta,alpha,iters = 10)


if __name__ == '__main__':
    main()
