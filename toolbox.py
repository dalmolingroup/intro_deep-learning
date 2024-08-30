import numpy as np
from tensorflow import keras

def arruma_dados(df):

    '''
    This function divides a dataframe into train (85%) and test (15%) arrays 
    for naive example, i.e., there is no feature engineering!
    '''

    i = np.random.permutation(len(df))
    
    df = df.iloc[i, :]

    df['label'] = df['class'].apply(lambda x: 1.0 if x == 'Abnormal' else 0.0)
    
    X = df.iloc[:, 0:6].to_numpy()
    y = df.iloc[:, -1:].to_numpy()
    
    N_samples, N_features = X.shape
    N_train = int(0.85*N_samples)
    N_test = N_samples - N_train
    
    X_train = X[:N_train]
    y_train = y[:N_train]
    
    X_test = X[N_train:]
    y_test = y[N_train:]

    return (X_train, y_train), (X_test, y_test), df


def MyModel():

    In = keras.Input(shape=(6,))
        
    x = keras.layers.Dense(50, activation='sigmoid')(In)
    
    x = keras.layers.Dense(200, activation='sigmoid')(x)
    
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(20, activation='sigmoid')(x)
    
    Out = keras.layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs=In, outputs=Out)




