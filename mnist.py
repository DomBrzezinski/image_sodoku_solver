import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model, load_model
import random
import cv2
from cropPuzzle import wait_q

def train():
    np.random.seed(0)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape[0])
    assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
    assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
    assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
    num_of_samples = []
    
    cols = 5
    num_classes = 10
    
    fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 8))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j))
                num_of_samples.append(len(x_selected))
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the training dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    X_train = X_train/255 
    X_test = X_test/255
    
    num_pixels = 784
    X_train = X_train.reshape(X_train.shape[0], num_pixels)
    X_test = X_test.reshape(X_test.shape[0], num_pixels)
    def create_model():
        model = Sequential()
        model.add(Dense(10, input_dim=num_pixels, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    
    model = create_model()
    print(model.summary())
    
    history = model.fit(X_train, y_train, validation_split=0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)

    model.save('my_model.keras')

def load(model_name):
    return load_model(model_name) 

def predict(img, model):
    
    img = np.asarray(img)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    T, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    full_image = img
    
    img = img/255
    img = img.reshape(1, 784)
    
    prediction = np.argmax(model.predict(img),axis=1)
    print("predicted digit:", str(prediction))

    cv2.imshow("full_image", cv2.resize(full_image, (500,500)))
    cv2.resizeWindow("full_image", 500, 500)
    wait_q()
