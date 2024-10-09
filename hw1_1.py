import numpy as np
import pandas as pd
import math
from tqdm import tqdm


def standardization(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data-mu)/sigma

def sum_of_square_error(y_predict, y_true):
    X = (y_true - y_predict)**2
    return np.sum(X)

def sigmoid(X):
    X = np.asarray(X,dtype=np.float64)
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X)*(1-sigmoid(X))

# import and preprocess data
df = pd.read_csv('2024_energy_efficiency_data.csv')
# one hot encoding
df = pd.get_dummies(df, columns=['Orientation','Glazing Area Distribution'])
df = df.drop(labels='Cooling Load',axis=1)
cols = ['# Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
 'Overall Height', 'Glazing Area']
df[cols] = df[cols].apply(standardization)
# shuffle and split
training_set = df.sample(frac=0.75, random_state=313552034)
testing_set = df.drop(training_set.index)
# separate features and labels
X_training_set = training_set.drop(labels='Heating Load',axis=1).to_numpy()
Y_training_set = training_set['Heating Load'].to_numpy().reshape(-1,1)
X_testing_set = testing_set.drop(labels='Heating Load',axis=1).to_numpy()
Y_testing_set = testing_set['Heating Load'].to_numpy().reshape(-1,1)

EPOCH = 10000
learning_rate = 0.001
np.random.seed(313552034)
W1 = np.random.uniform(low=-1, high=1, size=(16, 10))
W2 = np.random.uniform(low=-1, high=1, size=(10, 1))



mse = np.zeros(EPOCH)
for i in tqdm(range(EPOCH),desc="Loading", unit="iteration"):
    ran = np.random.randint(len(Y_training_set))
    This_X_training_set = X_training_set[ran].reshape(1,-1)
    This_Y_training_set = Y_training_set[ran].reshape(1,-1)

    # Layer 1
    Z1 = np.dot(This_X_training_set,W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    # Layer 2
    Z2 = np.dot(Z1,W2)
    # Error
    mse[i] = sum_of_square_error(This_Y_training_set,Z2)
    if i % (EPOCH//10) == 0: tqdm.write(f"Iteration {i}, MSE: {mse[i]}")  

    #back propagation
    D2 = 2 * (Z2 - This_Y_training_set)
    dW2 = np.dot(A1.T, D2)
    D1 = np.dot(D2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(This_X_training_set.T, D1)

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2


print("Training E_MRS: ",math.sqrt(mse[EPOCH-1]/len(Y_training_set)))
A1=np.dot(X_testing_set,W1)
A2=np.dot(A1,W2)
mse =  (Y_testing_set - A2)**2
mse_2 = math.sqrt(np.sum(mse)/len(Y_testing_set))
print("Testing E_MRS: ",mse_2)

