from warnings import filterwarnings
from tqdm import tqdm
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
TRAIN_SHAPE = (-1,137,236,1)

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

def save_score(y,y_pred):
    scores = dict(
        accuracy = accuracy_score(y,y_pred),
        precision = precision_score(y,y_pred),
        recall = recall_score(y,y_pred),
    )
    print (scores)
    json.dump(open("./score.json","w+"),scores)

def load_data(file_id):
    images = pd.read_parquet(f"./train/train_image_data_{file_id}.parquet")
    _labels = labels.loc[images.pop("image_id")]['grapheme_root']

    return images,_labels    

labels = pd.read_csv("./train.csv")
labels.index = labels.image_id

le_grapheme_root = LabelEncoder()
le_grapheme_root.fit(labels.grapheme_root)
labels['grapheme_root'] = le_grapheme_root.transform(labels.grapheme_root)



EPOCHS = 20
BATCH_SIZE = 5

model = create_vgg16(input_shape=(137,236,1),output_shape=len(list(set(labels.grapheme_root))))
model.compile(optimizer=OPTIMIZER,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

for epoch in range(EPOCHS):
    for file_id in range(4):
        X,Y = load_data(file_id)
        for index in tqdm(range(BATCH_SIZE,X.shape[0],BATCH_SIZE)):
            X_ = X[index - BATCH_SIZE:index].values.reshape(TRAIN_SHAPE)
            Y_ = Y[index - BATCH_SIZE:index]
            model.fit(X_,Y_,batch_size=BATCH_SIZE,verbose=False)

        X_ = X[index:]
        Y_ = Y[index:]
        model.fit(X_,Y_)

# model.evaluate(x,y)
# save_score(model,x,y)

notify.success()

for i in globals():
    del globals()[i]

exit()