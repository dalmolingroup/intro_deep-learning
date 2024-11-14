from tensorflow import keras

def MyModel():

    In = keras.Input(shape=(6,))
        
    x = keras.layers.Dense(50, activation='sigmoid')(In)
    
    x = keras.layers.Dense(200, activation='sigmoid')(x)
    
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(20, activation='sigmoid')(x)
    
    Out = keras.layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs=In, outputs=Out)

