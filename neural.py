import numpy as np
import math
import random
from matplotlib import pyplot as plt

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
    theta_1 = {10:[],11:[],12:[],20:[],21:[],22:[]}
    theta_2 = {10:[],11:[],12:[],20:[],21:[],22:[]}
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
        #J_theta.append((-1/m)*np.sum((y * np.log(outputlayer_outputs) + (1-y) * np.log(1-outputlayer_outputs))))
        J_theta.append(0.5 * np.sum((y-outputlayer_outputs)**2))
        p.append(i)



        # for output layer


        # backward propogation

        # calculate error for last layer
        Error_output = outputlayer_outputs - y


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

        #appending theta values for plotting
        theta_1[10].append(bias_hidden[0][0])
        theta_1[11].append(theta[1][0][0])
        theta_1[12].append(theta[1][0][1])
        theta_1[20].append(bias_hidden[0][1])
        theta_1[21].append(theta[1][1][0])
        theta_1[22].append(theta[1][1][1])

        theta_2[10].append(bias_output[0][0])
        theta_2[11].append(theta[2][0][0])
        theta_2[12].append(theta[2][0][1])
        theta_2[20].append(bias_output[0][1])
        theta_2[21].append(theta[2][1][0])
        theta_2[22].append(theta[2][1][1])

    #plotting the cost function for each iteration
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_title('Cost function plot')
    ax.plot(p, J_theta, '.', color='blue')
    plt.show()

    # plotting the theta(1) values for each iteration
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Theta(1) values')
    ax.plot(p,theta_1[10] , '.', color='blue',label = 'theta(1) 10')
    ax.plot(p, theta_1[11], '.', color='red',label = 'theta(1) 11')
    ax.plot(p, theta_1[11], '.', color='yellow',label = 'theta(1) 12')
    ax.plot(p, theta_1[20], '.', color='green',label = 'theta(1) 20')
    ax.plot(p, theta_1[21], '.', color='magenta',label = 'theta(1) 21')
    ax.plot(p, theta_1[22], '.', color='orange',label = 'theta(1) 22')
    ax.legend()
    plt.show()

    # plotting the theta(2) values for each iteration
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title('Theta(2) values')
    ax.plot(p, theta_2[10], '.', color='blue', label='theta(2) 10')
    ax.plot(p, theta_2[11], '.', color='red', label='theta(2) 11')
    ax.plot(p, theta_2[11], '.', color='yellow', label='theta(2) 12')
    ax.plot(p, theta_2[20], '.', color='green', label='theta(2) 20')
    ax.plot(p, theta_2[21], '.', color='magenta', label='theta(2) 21')
    ax.plot(p, theta_2[22], '.', color='orange', label='theta(2) 22')
    ax.legend()
    plt.show()

    return outputlayer_outputs



def main():
    X = np.array([[0.05,0.1]])
    inp_dim = len(X[0])
    num_of_samples = len(X)
    num_of_outputs = 2
    hidden_n = 2
    iterations = 100000

    # Output
    y = np.array([[0.01,0.99]])

    #initialize theta values to random and uniformly distribited numbers from 0 to 1
    theta = {1: np.random.uniform(size=(inp_dim,hidden_n)), 2: np.random.uniform(size=(hidden_n,num_of_outputs))}

    #initialize bias to 1
    bias_hidden = np.ones((1,hidden_n))
    bias_output = np.ones((1,num_of_outputs))
    #theta = {1:np.array([[-30,20,20],[10,-20,-20]]),2:np.array([[-10,20,20]])}
    print theta[1].shape
    print theta[2].shape

    answers = classify(iterations, hidden_n, num_of_samples, X, theta, num_of_outputs, y,bias_hidden,bias_output)
    print answers



if __name__ == '__main__':
    main()


