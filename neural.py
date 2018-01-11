#Author: Ved Paranjape
#Neural Network
# The following is a neural network for classifying iris data. The parameters under consideration are petal length and
# petal width. The network is trained using leave-one-out analysis. The dataset consists of 100 samples. The training
# is done using 99 samples and the remaining sample is used for testing. This process is repeated. The model works with
# an accuracy of 94 %.

import numpy as np
from sklearn import datasets


def test_classify(hidden_n,X, theta, output_vars,bias_hidden,bias_output):
    # forward propogation

    # for hidden layer
    hiddenlayer_nobias = np.dot(X, theta[1])
    # adding bias to input
    hiddenlayer_inputs = hiddenlayer_nobias + bias_hidden
    hiddenlayer_outputs = sigmoid(hiddenlayer_inputs)

    # for output layer
    outputlayer_inputs_nobias = np.dot(hiddenlayer_outputs, theta[2])

    # adding bias to output
    outputlayer_inputs = outputlayer_inputs_nobias + bias_output
    outputlayer_outputs = sigmoid(outputlayer_inputs)

    return outputlayer_outputs[0]

#sigmoid derivative function
def sigmoid_derivative(a):
    return a * (1 - a)

#sigmoid function same as logistic regression
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def classify(iters, hidden_n, num_of_samples, X, theta, output_vars, y,bias_hidden,bias_output):
    alpha = 0.1
    J_theta = []
    m = len(X)
    p = []
    for i in range(iters):

        # forward propogation

        # for hidden layer
        hiddenlayer_nobias = np.dot(X, theta[1])
        #adding bias to input
        hiddenlayer_inputs = hiddenlayer_nobias + bias_hidden
        hiddenlayer_outputs = sigmoid(hiddenlayer_inputs)

        #for output layer
        outputlayer_inputs_nobias = np.dot(hiddenlayer_outputs,theta[2])

        #adding bias to output
        outputlayer_inputs = outputlayer_inputs_nobias + bias_output
        outputlayer_outputs = sigmoid(outputlayer_inputs)

        #calculating cost function values
        J_theta.append((-1/m)*np.sum((y * np.log(outputlayer_inputs) + (1-y) * np.log(1-outputlayer_inputs))))
        #J_theta.append(0.5 * np.sum((y-outputlayer_outputs)**2))
        p.append(i)

        # backward propogation

        # calculate error for last layer
        Error_output = outputlayer_outputs - y
        #Error_output = (y * np.log(outputlayer_inputs) + (1-y) * np.log(1-outputlayer_inputs))


        # calculating derivative of output layer activations and hidden layer activations
        der_3 = sigmoid_derivative(outputlayer_outputs)
        der_2 = sigmoid_derivative(hiddenlayer_outputs)

        #calculate delta for output layer
        delta_output = Error_output * der_3

        #calculate delta for hidden layer
        Error_hidden = delta_output.dot(theta[2].T)

        # calculating delta for hidden layer
        delta_hidden = Error_hidden * der_2

        #update weights
        theta[2] = theta[2] - alpha * ((hiddenlayer_outputs.T.dot(delta_output)))
        theta[1] = theta[1] - alpha*(X.T.dot(delta_hidden))

        #update biases
        bias_hidden = bias_hidden -  np.sum(delta_hidden, axis=0, keepdims=True)*alpha
        bias_output = bias_output - np.sum(delta_output, axis=0, keepdims=True)*alpha

    return bias_hidden,bias_output,theta



def main():
    iris = datasets.load_iris()
    X = iris.data[50:,2:]
    X[:,0] = (X[:,0] - np.min(X[:,0])) / (np.max(X[:,0]) - np.min(X[:,0]))
    X[:, 1] = (X[:, 1] - np.min(X[:, 1])) / (np.max(X[:, 1]) - np.min(X[:, 1]))
    X1 = X
    y = iris.target[50:]
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        if y[i] == 2:
            y[i] = 1
    # X = np.array([[0.05,0.1]])
    inp_dim = len(X[0])
    y = y.reshape(100,1)
    y1 = y
    error = 0

    #running the neural network 100 times using leave-one-out-analysis
    for test in range(100):
        print ("Test "),test+1
        X = X1
        y = y1
        X_test = X[test]
        Y_test = y[test]
        X = np.delete(X,test,0)
        y = np.delete(y,test,0)
        num_of_samples = len(X)
        num_of_outputs = 1
        hidden_n = 2
        iterations = 50000

        #initialize theta values to random and uniformly distribited numbers from 0 to 1
        theta = {1: np.random.uniform(size=(inp_dim,hidden_n)), 2: np.random.uniform(size=(hidden_n,num_of_outputs))}

        #initialize bias to 1
        bias_hidden = np.ones((1,hidden_n))
        bias_output = np.ones((1,num_of_outputs))

        bias_hidden,bias_output,theta = classify(iterations, hidden_n, num_of_samples, X, theta, num_of_outputs, y,bias_hidden,bias_output)
        prob = test_classify(hidden_n, X_test, theta, num_of_outputs,bias_hidden, bias_output)
        for i in prob:
            if(i<0.5):
                answers = 0
            else:
                answers = 1
        print str(answers) + "\t" + str(Y_test[0]) + "\t" + str(prob)
        if(answers != Y_test[0]):
            error = error + 1
    print error
    average_error_rate = (error / 100.0)
    print ("Average Error Rate:"),average_error_rate



if __name__ == '__main__':
    main()

