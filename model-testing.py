from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2

model = load_model('vgg-car-body-type-final.h5')

path = "results"

for img in os.listdir(path):
    img = image.load_img(path + "/" + img, target_size=(224,224))
    plt.imshow(img)
    plt.show()

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=1)

    category = ""
    if pred[0][0] > 0.5:
        category = "Cab"
    elif pred[0][1] > 0.5:
        category = "Convertible"
    elif pred[0][2] > 0.5:
        category = "Coupe"
    elif pred[0][3] > 0.5:
        category = "Hatchback"
    elif pred[0][4] > 0.5:
        category = "Minivan"
    elif pred[0][5] > 0.5:
        category = "SUV"
    elif pred[0][6] > 0.5:
        category = "Sedan"
    elif pred[0][7] > 0.5:
        category = "Van"
    elif pred[0][8] > 0.5:
        category = "Wagon"

    print(category)