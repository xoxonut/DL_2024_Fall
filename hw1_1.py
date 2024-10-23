# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
np.random.seed(313552034)

def one_hot_encoding(df, cols):
    cols_to_encode = [col for col in cols if col in df.columns]
    return pd.get_dummies(df, columns=cols_to_encode)

def standardization(df,cols):
    def standardize(dtaa):
        mean = np.mean(dtaa, axis=0)
        std = np.std(dtaa, axis=0)
        return (dtaa - mean) / std
    return df[cols].apply(standardize)

def train_test_split(df, train_szie):
    train_set = df.sample(frac=train_szie, random_state=313552034)
    test_set = df.drop(train_set.index)
    return train_set, test_set

def sep_feature_label(df, label):
    X = df.drop(label, axis=1).to_numpy()
    Y = df[label].to_numpy().reshape(-1,1)
    return X,Y


def sum_of_square_error(y_predict, y_true):
    X = (y_true - y_predict)**2
    return np.sum(X)

def ReLU(X):
    return np.maximum(0,X)

def dReLU(X):
    return np.where(X > 0, 1, 0)

def training(EPOCH,learning_rate,input_width,hidden_width,X,Y):
    # initialize weights
    W1 = np.random.uniform(-1,1,(input_width,hidden_width))
    B1 = np.random.uniform(-1,1,(1,hidden_width))
    W2 = np.random.uniform(-1,1,(hidden_width,1))
    B2 = np.random.uniform(-1,1,(1,1))
    mse = np.zeros(EPOCH)
    # Do not use sigomid activation function for the hidden layer
    order = np.arange(len(X))
    for i in range(EPOCH):
        np.random.shuffle(order)
        for stochastic in order:
            stochastic_X = X[stochastic].reshape(1,-1)   
            stochastic_Y = Y[stochastic].reshape(1,-1)
            # Layer 1
            Z1 = np.dot(stochastic_X,W1)
            Z1 = Z1 + B1
            Z1 = ReLU(Z1)
            # Layer 2   
            Z2 = np.dot(Z1,W2)
            Z2 = Z2 + B2
            # Backpropagation
            dB2 = 2*(Z2-stochastic_Y)
            dW2 = np.dot(Z1.T,dB2)
            tmp = np.dot(dB2,W2.T)
            dB1 = tmp*dReLU(Z1)
            dW1 = np.dot(stochastic_X.T,dB1)
            # Update weights
            W1 = W1 - learning_rate*dW1        
            B1 = B1 - learning_rate*dB1
            W2 = W2 - learning_rate*dW2
            B2 = B2 - learning_rate*dB2
        Z1 = np.dot(X,W1)
        Z1 = Z1 + B1
        Z1 = ReLU(Z1)
        Z2 = np.dot(Z1,W2)
        Z2 = Z2 + B2
        mse[i] = sum_of_square_error(Y,Z2)
    return W1,W2,B1,B2,mse
def prediction(X,W1,W2,B1,B2):
    Z1 = np.dot(X,W1)
    Z1 = Z1 + B1
    Z1 = ReLU(Z1)   
    Z2 = np.dot(Z1,W2)
    Z2 = Z2 + B2
    return Z2

def root_mean_square_error(Y_predict,Y_true):
    return math.sqrt(sum_of_square_error(Y_predict,Y_true)/len(Y_true))

def plt_learning_curve(mse,EPOCH):
    num_epoch = np.arange(0,EPOCH)
    
    # plt.ylim(0,np.max(mse))
    plt.xlim(0,EPOCH)
    plt.xlabel('epoch')
    plt.ylabel('Error')
    plt.plot(num_epoch,mse)
    plt.title("Learning curve")
    plt.savefig('Learning curve.png')

def plt_predict_train(X_train_set,Y_train_set,W1,W2,B1,B2):
    plt.clf()
    num_train_set = np.arange(0,len(Y_train_set))
    plot_train_predict = prediction(X_train_set,W1,W2,B1,B2)
    plot_train_true = Y_train_set
    plt.xlabel('# th case')
    plt.ylabel('heating load')
    plt.plot(num_train_set,plot_train_predict , label = "predict",linewidth='0.5')
    plt.plot(num_train_set,plot_train_true , label = "label",linewidth='0.5')
    plt.title("Prediction for train data")
    plt.legend()
    plt.savefig('Prediction for train data.png')

def plt_predict_test(X_test_set,Y_test_set,W1,W2,B1,B2):    
    plt.clf()
    num_test_set = np.arange(0,len(Y_test_set))
    plot_test_predict = prediction(X_test_set,W1,W2,B1,B2)
    plot_test_true = Y_test_set
    
    plt.xlabel('# th case')
    plt.ylabel('heating load')
    plt.plot(num_test_set,plot_test_predict ,  label = "predict",linewidth='0.5')
    plt.plot(num_test_set,plot_test_true , label = "label",linewidth='0.5')
    plt.title("Prediction for test data")
    plt.legend()
    plt.savefig('Prediction for test data.png')



# %%
EPOCH = 10000
learning_rate = 0.0005
df = pd.read_csv('2024_energy_efficiency_data.csv')
df = df.drop(labels='Cooling Load',axis=1)
cols = ['# Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
        'Overall Height', 'Glazing Area']
df[cols] = standardization(df, cols)
df1 = df.copy()
df = one_hot_encoding(df, ['Orientation', 'Glazing Area Distribution'])
train_set, test_set = train_test_split(df, 0.75)
X_train_set , Y_train_set = sep_feature_label(train_set, 'Heating Load')
X_test_set , Y_test_set = sep_feature_label(test_set, 'Heating Load')
W1,W2,B1,B2,mse = training(EPOCH,learning_rate,X_train_set.shape[1],10,X_train_set,Y_train_set)
mrs_train = root_mean_square_error(prediction(X_train_set,W1,W2,B1,B2),Y_train_set)
mrs_test = root_mean_square_error(prediction(X_test_set,W1,W2,B1,B2),Y_test_set)
print('Root mean square error of training set:',mrs_train)
print('Root mean square error of testing set:',mrs_test)

# %%
plt_learning_curve(mse,EPOCH)
plt_predict_train(X_train_set,Y_train_set,W1,W2,B1,B2)
plt_predict_test(X_test_set,Y_test_set,W1,W2,B1,B2)

# %%
df_without = {i: df1.drop(i, axis=1) for i in df1.columns if i != 'Heating Load'}
for key,value in df_without.items():
    value = one_hot_encoding(value, ['Orientation', 'Glazing Area Distribution'])
    train_set, test_set = train_test_split(value, 0.75)
    X_train_set , Y_train_set = sep_feature_label(train_set, 'Heating Load')
    X_test_set , Y_test_set = sep_feature_label(test_set, 'Heating Load')
    W1,W2,B1,B2,mse = training(EPOCH,learning_rate,X_train_set.shape[1],10,X_train_set,Y_train_set)
    mrs_train = root_mean_square_error(prediction(X_train_set,W1,W2,B1,B2),Y_train_set)
    mrs_test = root_mean_square_error(prediction(X_test_set,W1,W2,B1,B2),Y_test_set)
    print(key,'Root mean square error of training set:',mrs_train)
    print(key,'Root mean square error of testing set:',mrs_test)
    plt_learning_curve(mse,EPOCH)
    plt_predict_train(X_train_set,Y_train_set,W1,W2,B1,B2)
    plt_predict_test(X_test_set,Y_test_set,W1,W2,B1,B2)
    print('==================================================')


