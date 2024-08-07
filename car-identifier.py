# Importing the necessary libraries, let's see if anything goes wrong
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

IMAGE_SIZE = [224, 224]
# Paths for all the training, testing and validation folders
train_path = "body-type-dataset/train"
test_path = "body-type-dataset/test"
validation_path = "body-type-dataset/validation"

# Numpy arrays
x_train = []

for folder in os.listdir(train_path):
    train_sub_paths = train_path + "/" + folder

    for image in os.listdir(train_sub_paths):
        train_image_path = train_sub_paths + "/" + image
        train_image_array = cv2.imread(train_image_path)
        train_image_array = cv2.resize(train_image_array,(224, 224))
        x_train.append(train_image_array)


x_test = []

for folder in os.listdir(test_path):
    test_sub_paths = test_path + "/" + folder

    for image in os.listdir(test_sub_paths):
        test_image_path = test_sub_paths + "/" + image
        test_image_array = cv2.imread(test_image_path)
        test_image_array = cv2.resize(test_image_array, (224, 224))
        x_test.append(test_image_array)

x_val = []

for folder in os.listdir(validation_path):
    validation_sub_paths = validation_path + "/" + folder

    for image in os.listdir(validation_sub_paths):
        validation_image_path = validation_sub_paths + "/" + image
        validation_image_array = cv2.imread(validation_image_path)
        validation_image_array = cv2.resize(validation_image_array, (224, 224))
        x_val.append(validation_image_array)

print("Loaded the arrays")

train_x = np.array(x_train)
test_x = np.array(x_test)
validation_x = np.array(x_val)

print ("Created the np arrays")

# Rescaling images
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)

print("Scaled the images")

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size= (224, 224),
                                                 batch_size= 32,
                                                 class_mode= 'sparse')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size= (224, 224),
                                            batch_size= 32,
                                            class_mode= 'sparse')

validation_set = validation_datagen.flow_from_directory(validation_path,
                                                        target_size= (224, 224),
                                                        batch_size= 32,
                                                        class_mode= 'sparse')
print("Created the sets")

# Creating the training set class arrays
train_y = training_set.classes
test_y = test_set.classes
validation_y = validation_set.classes


# Model building?
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(9, activation='softmax')(x)

# Creating a model object
model = Model(inputs=vgg.input, outputs=prediction)

# Printing the model summary?
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=5)

# Training the model
history = model.fit(
    train_x,
    train_y,
    validation_data=(validation_x, validation_y),
    epochs=10,
    callbacks=[early_stop],
    batch_size=32,
    shuffle=True
)

# The losses:
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('vgg-loss-car-body-1.png')
plt.show()

# Accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('vgg-acc-car-body-1.png')
plt.show()

# Evaluating the model
model.evaluate(test_x, test_y, batch_size=32)

# Making a prediction and then determining if it is equal to the test results
y_pred = model.predict(test_x).round()
y_pred = np.argmax(y_pred,axis=1)

# Printing the accuracy score
accuracy_score(y_pred, test_y)

print(classification_report(y_pred, test_y))

confusion_matrix(y_pred, test_y)

# Saving the model
model.save('vgg-car-body-type-final.h5')
