import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

def carregar_mnist(reformar=False, embaralhar=False, separar=False):
    
    # 1. Importar dados do MNIST:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
    
    # 2. Embaralhar dados:
    if embaralhar:
        # dados para treinamento:
        i = np.random.permutation(x_train.shape[0])
        
        x_train = x_train[i]
        y_train = y_train[i]

        # dados para teste:
        i = np.random.permutation(x_test.shape[0])
    
        x_test = x_test[i]
        y_test = y_test[i]

    # 3. Reformar dados:
    if reformar:
        # 3.1 Normalizar images:
        Max = x_train.max()
        
        x_train = x_train/Max
        x_test = x_test/Max

        # 3.2 Aplainar matrizes para vetores:
        def flatten(x):
            N, n, m = x.shape
            return x.reshape(N, n*m)
        
        x_train = flatten(x_train)
        x_test = flatten(x_test)

        # 3.3 Codificação one-hot para os dados rótulos:
        onehot = layers.CategoryEncoding(num_tokens=10, output_mode="one_hot")
        
        y_train = onehot(y_train).numpy()
        y_test = onehot(y_test).numpy()

    else:
        i = 0
        plt.figure(figsize=(5,3))
        plt.title("Number "+str(y_train[i]))
        plt.imshow(x_train[i], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    print(f"x-train:{x_train.shape}, y-train:{y_train.shape}")
    print(f"x-test:{x_test.shape},  y-test:{y_test.shape}")

    return (x_train, y_train), (x_test, y_test)


def MeuModelo():
        
    In = keras.Input(shape=(28*28,))
    
    x = layers.Dense(300, activation="relu")(In)
    
    x = layers.Dense(100, activation="relu")(x)
    
    Out = layers.Dense(10, activation="softmax")(x)
    
    return keras.Model(inputs=In, outputs=Out)

