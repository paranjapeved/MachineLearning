import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import datasets
from sklearn import linear_model
import random


# beta_0 = -4.0
# beta_1 = 1.4
# sigma = 0.25

def do_linear(x,y,n):
    #beta_1_hat = np.sum((x-np.mean(x)*(y-np.mean(y)))/np.sum(x-(np.mean(x)**2)))
    beta_1_hat = (np.sum(x*y) - n*np.mean(x)*np.mean(y))/((np.sum(x**2)) - n*(np.mean(x)**2))
    #beta_1_hat = np.cov(x,y)[1][0]/np.var(x)
    beta_0_hat = np.mean(y) - beta_1_hat*np.mean(x)
    y_hat = beta_0_hat + beta_1_hat * x
    return (beta_0_hat,beta_1_hat,y_hat)

def main():

    #creating variables for testing and training data
    x_test = []
    y_test = []

    #load diabetes input data
    diabetes = datasets.load_diabetes()
    x = diabetes.data[:,2]
    y = diabetes.target
    #beta_0_hat,beta_1_hat,y_hat = do_linear(x,y,len(x))

    # x_test, y_test are test data variables

    lin_regression = linear_model.LinearRegression()
    x = x.reshape((len(x), 1))
    print x.shape
    y = y.reshape((len(y), 1))
    print y.shape

    #creating test data from training data
    for i in range(20):
        index = random.randint(0,len(x))
        x_test.append(x[index])
        np.delete(x,index,0)
        y_test.append(y[index])
        np.delete(y,index,0)

    # fit the linear regression model to x and y(training data variables
    lin_regression.fit(x,y)
    #predict y values using the constructed model
    y_hat = lin_regression.predict(x_test)

    #plot x_test vs predicted y and x_test vs y_test
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_test,y_hat,'.',color='blue',label = 'testing x VS predicted y')
    ax.plot(x_test,y_test,'.',color='red',label = 'testing x VS testing y')
    ax.legend()
    plt.show()
    #R_sq,SSR,SSE,n = rsquare(x,y,y_hat)
    #p = calculate_pvalue(SSR,SSE,n)

def rsquare(x,y,y_hat):
    y_mean = np.mean(y)
    n = len(x)
    sigma_hat = np.sqrt(np.sum((y-y_hat)**2))/(n-2)
    #calculating the sum of squares
    SSR = np.sum((y_hat - y_mean)**2)
    SSE = np.sum((y - y_hat)**2)
    SST = SSE + SSR
    #find R2
    R_sq = SSR / SST
    return (R_sq,SSR,SSE,n)

def calculate_pvalue(SSR,SSE,n):
    #calculating the mean squares, F, p_value
    MS_reg = SSR
    MS_err = SSE / (n-2)
    print MS_reg
    print SSE
    print MS_err
    F = MS_reg/MS_err
    print ("F:"),F
    p_value = 1 - st.f._cdf(F,dfn=1,dfd=n-2)
    print ("pvalue:"),p_value
    return p_value

if __name__ == "__main__":
    main()
