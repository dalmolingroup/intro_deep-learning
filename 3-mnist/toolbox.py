import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def load_mnist():
    
    # import MNIST data:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
    
    # shuffling train data:
    i = np.random.permutation(x_train.shape[0])
    
    x_train = x_train[i]
    y_train = y_train[i]
    
    # shuffling test data:
    i = np.random.permutation(x_test.shape[0])
    
    x_test = x_test[i]
    y_test = y_test[i]
    
    # normalization:
    Max = x_train.max()
    
    x_train = x_train/Max
    x_test = x_test/Max
    
    # one-hot encoding for label data:
    def one_hot(labels):
        N = labels.size
        
        y_hot = np.zeros((N, 10), dtype="float32")
        
        for i, y in enumerate(labels):
            y_hot[i][y] = 1
    
        return y_hot
    
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    def flatten(x):
        N, n, m = x.shape
        return x.reshape(N, n*m)
    
    x_train = flatten(x_train)
    x_test = flatten(x_test)
        
    # splitting the train data into train and validation:
    N_val = int(0.2*x_train.shape[0])
    
    x_val = x_train[:N_val]
    x_train = x_train[N_val:]
    
    y_val = y_train[:N_val]
    y_train = y_train[N_val:]

    print(f"x-train:{x_train.shape}, x-val:{x_val.shape}, x-test:{x_test.shape}")
    print(f"y-train:{y_train.shape}, y-val:{y_val.shape}, y-test:{y_test.shape}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    

def dense_mnist():

    In = keras.Input(shape=(28*28,))
    
    x = layers.Dense(400, activation="relu")(In)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(200, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(100, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    
    Out = layers.Dense(10, activation="softmax")(x)
    
    return keras.Model(inputs=In, outputs=Out)

