import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Feature_scaling(X):
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X

def relu(Z):
    return np.maximum(Z, 0)

def relu_backward(dA, Z):

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    return (np.exp(Z) / np.sum(np.exp(Z), axis=0))

def softmax_backward(Z):
    Sz = softmax(Z)
    dZ = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return dZ

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    S = sigmoid(Z)
    dZ = dA * S * (1 - S)
    return dZ

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
       parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
       parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
       
    return parameters

def linear_activation_forward(A_prev, W, b, activation):

    Z = np.dot(W, A_prev) + b
    if activation == "softmax":
        A = softmax(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)   
    elif activation == "relu":
        A = relu(Z)

    cache = ((A, W, b), Z)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
            
    return AL, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = - 1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))   
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, Z = cache
    if activation == "relu":
        dZ = relu_backward(dA, Z)
    elif activation == "softmax":
        dZ = softmax_backward(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape) 
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters) 
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
def main():
    df = pd.read_csv('data.csv', header=None)
    Y = np.array(df[1])
    Y = np.where(Y == 'B', 0, 1)
    X = np.array(df.iloc[:, 2:])
    X = Feature_scaling(X)
    X = X.T
    Y = Y.reshape((len(Y), 1))
    layers_dims = [X.shape[0], 4, 3, 1]
    parameters = L_layer_model(X, Y, layers_dims, num_iterations = 900, print_cost = True)
    
if __name__ == "__main__":
    main();

