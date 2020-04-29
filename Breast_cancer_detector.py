import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
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
    return (np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    S = sigmoid(Z)
    dZ = dA * S * (1 - S)
    return dZ

def initialize_parameters(layer_dims):
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
    elif activation == "softmax":
        A = softmax(Z)
    cache = ((A_prev, W, b), Z)
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
    return AL, caches

def cross_entropy(AL,Y):
    cost = -np.mean(Y * np.log(AL.T + 1e-8))
    return cost

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), keepdims=True, axis=1)
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, AL, Y, cache, activation):
    linear_cache, Z = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "softmax":
        dZ = AL - Y.T
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    #Y = Y.reshape(AL.shape) 
    #dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(None, AL, Y, current_cache, 'softmax')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], None, None, current_cache, 'relu')
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

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    np.random.seed(1)
    costs = []
    parameters = initialize_parameters(layers_dims)
    for i in tqdm(range(0, num_iterations)):
        
        AL, caches = L_model_forward(X, parameters) 
        cost = cross_entropy(AL, Y)
        #cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1 == 0:
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
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = X_train.T
    #y_train = y_train.reshape((1, len(y_train)))
    X_test = X_test.T
    #y_test = y_test.reshape((1, len(y_test)))
    enc = OneHotEncoder(sparse=False, categories='auto')
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1))
    y_test = enc.transform(y_test.reshape(len(y_test), -1))
    layers_dims = [X_train.shape[0], 40, 20, 10, 5, 2]
    parameters = L_layer_model(X_train, y_train, layers_dims, num_iterations = 10000, print_cost = True)
    Yhat, _ = L_model_forward(X_test, parameters)
    Yhat = np.where(Yhat < 0.5, 0, 1)
    Yhat = Yhat.T
    result = (Yhat == y_test).mean()
    print("Accuracy : " + str(result * 100) + ' %')
if __name__ == "__main__":
    main();

