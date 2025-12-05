import tensorflow as tf
from tensorflow.keras import layers, models

#imp notes:
# In CIFAR-10 dataset, images are of size 32x32 pixels with 3 color channels (RGB) and there are 10 classes to classify.
# The model architecture consists of multiple convolutional layers followed by batch normalization, max pooling, and dropout layers to prevent overfitting.
# Finally, the model is flattened and connected to a dense layer for classification.



def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()
    
    # 1st Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # 2nd Block (mainly i have added padding to retain spatial dimensions and multiple conv layers for better feature extraction)
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # 3rd Block (added for more accurate feature learning and extraction of various features)
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    # Final connection of all layers 
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))    
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # The Final Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model