import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCH = 1000
LEARNING_RATE = 0.0005
HIDDEN_LAYER_WIDTH = 20
np.random.seed(313552034)

df = pd.read_csv('2024_ionosphere_data.csv', names=list(range(35)))
df[34] = df[34].map({'g': 1, 'b': 0})
label = df.pop(34)
label = pd.get_dummies(label,prefix=34)
df = pd.concat([label, df], axis=1)
# split the data into training and testing
train = df.sample(frac=0.8)
test = df.drop(train.index)
train = train.to_numpy()
test = test.to_numpy()
# split the data into featrues (X) and labels (Y)
X_train = np.delete(train, [0, 1], axis=1)
Y_train = train[:, 0:2].reshape(-1, 2)
X_test = np.delete(test, [0, 1], axis=1)
Y_test = test[:, 0:2].reshape(-1, 2)


def sigmoid(x):
    x = np.array(x,dtype=np.float128)
    x.clip(-500,500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    x = np.array(x,dtype=np.float128)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_hat):
    return -np.sum(y_true * np.log(y_hat))

def predict(X,W1,W2):
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = softmax(Z2)
    return A2


def plt_learning_curve(loss,EPOCH):
    num_epoch = np.arange(0,EPOCH)
    plt.ylim(0,np.max(loss))
    plt.xlim(0,EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.plot(num_epoch,loss)
    plt.title("Learning curve")
    plt.savefig('Learning curve_2.png')

order = np.arange(X_train.shape[0])
cel =[]
def train():
    W1 = np.random.normal(0,0.5,(34,HIDDEN_LAYER_WIDTH))
    W2 = np.random.normal(0,0.5,(HIDDEN_LAYER_WIDTH,2))
    for i in tqdm(range(EPOCH)):
        np.random.shuffle(order)
        for stochastic in order:
            stochastic_X = X_train[stochastic].reshape(1, -1)
            stochastic_Y = Y_train[stochastic].reshape(1, -1)
            # Layer 1
            Z1 = np.dot(stochastic_X, W1)
            A1 = sigmoid(Z1)
            # Layer 2
            Z2 = np.dot(A1, W2)
            A2 = softmax(Z2)
            
            # Backpropagation
            dZ2 = A2 - stochastic_Y
            dW2 = np.dot(A1.T, dZ2)
            dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
            dW1 = np.dot(stochastic_X.T, dZ1)
            
            # Update weights
            W1 = W1 - LEARNING_RATE * dW1
            W2 = W2 - LEARNING_RATE * dW2
        Z1 = np.dot(X_train, W1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2)
        A2 = softmax(Z2)
        if i == EPOCH-1 or i == EPOCH//2 or i==1:
            for index in range(len(Z2)):
                color = 'b' if Y_train[index][0] == 1 else 'r'
                plt.scatter(Z2[index,0], Z2[index,1],
                            color=color)
            plt.title(f'Hidden width: {format(HIDDEN_LAYER_WIDTH)}, Epoch: {i}')
            plt.legend()
            plt.savefig(f'2_HIDDEN_LAYER_WIDTH: {HIDDEN_LAYER_WIDTH}_EPOCH: {i}.png')

        
        cel.append(cross_entropy_loss(Y_train, A2))
    return cel,W1,W2
def error_rate(Y_hat, Y):
    return np.mean(np.argmax(Y_hat, axis=1) != np.argmax(Y, axis=1))
cel,W1,W2 = train()
plt_learning_curve(cel,EPOCH)
print("Training error rate: ", error_rate(predict(X_train,W1,W2), Y_train)*100 ,"%")
print("Testing error rate: ", error_rate(predict(X_test,W1,W2), Y_test)*100 ,"%")