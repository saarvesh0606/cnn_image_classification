import os
import numpy as np
import tensorflow as tf  # FIXED: tensflow -> tensorflow
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import build_cnn 

os.makedirs('results', exist_ok=True)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


# I have loaded the data from cifar-10 dataset and normalized it between 0 and 1 
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


# In this function I have plotted training and validation accuracy curves 
def plot_training_curves(history):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_validation_accuracy.png')
    plt.close()
    
    
# in this function i have plotted training and validation loss curves
def plot_loss_curves(history):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='Train Loss') 
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')       
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_validation_loss.png')
    plt.close()
    print('Data and graphs are stored in the results directory.')
    

# Here is the main function where model is built, compiled, trained and saved for future evaluation    
def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f'Training data shape: {x_train.shape}')
    print(f'Testing data shape: {x_test.shape}')  
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )  
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    batch_size = 64
    epochs = 25 
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    model.save("results/cnn_cifar10_model.keras")
    print("Model saved to results/cnn_cifar10_model.h5")
    plot_training_curves(history)
    plot_loss_curves(history)
    
if __name__ == "__main__":
    main()