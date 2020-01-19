from warnings import filterwarnings
from tqdm import tqdm

try:
    from notifyme import notify
except:
    pass

filterwarnings("ignore")

import sys
import os

import pandas as pd 
import numpy as np 
import cv2 as cv
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,precision_score,accuracy_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model


PATH = os.getcwd()
CROP = True if "--crop" in sys.argv else False
RESIZE = True if "--resize" in sys.argv else False
SHAPE = (137,236)

OPTIMIZER = sys.argv[sys.argv.index("-opt")+1] if "-opt" in sys.argv else "adam"
ARCHITECTURE = sys.argv[sys.argv.index("-arch")+1] if "-arch" in sys.argv else "vgg16"

class LabelEncoder():
    def __init__(self,):
        self._classes = []

    def fit(self,labels):
        self._classes = list(set(labels))

    def transform(self,labels):
        return np.fromiter(map(self._classes.index,labels),dtype=np.int)

def create_vgg16(input_shape,output_shape):
    model = Sequential()

    ### Block 1
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    ## Block 2
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    ## Block 3
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    ## Block 4
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    ## Prediction Block
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=output_shape, activation="softmax"))

    return model

def save_score(model,x,y):
    y_pred = model.predict_classes(x)
    json.dump(open("./score.json","w+"),dict(
        accuracy = accuracy_score(y,y_pred),
        precision = precision_score(y,y_pred),
        recall = recall_score(y,y_pred),
    ))

images = pd.read_parquet("./train/train_image_data_0.parquet")
labels = pd.read_csv("./train.csv")
labels.index = labels.image_id
labels =  labels.loc[images.pop("image_id")]
images = images.values.reshape(-1,137,236,1)
output_shape = labels.grapheme.nunique()
grapheme_le = LabelEncoder()
grapheme_le.fit(labels.grapheme)
labels = grapheme_le.transform(labels.grapheme)
X,x,Y,y = train_test_split(images,labels)

del images,labels

model = create_vgg16(input_shape=X.shape[1:],output_shape=output_shape)
model.compile(optimizer=OPTIMIZER,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X,Y,epochs=10)
model.evaluate(x,y)

save_score(model,x,y)

notify.success()