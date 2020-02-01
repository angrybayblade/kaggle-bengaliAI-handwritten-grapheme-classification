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

from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


PATH = os.getcwd()
CROP = True if "--crop" in sys.argv else False
RESIZE = True if "--resize" in sys.argv else False

SHAPE = (137,236)
SHAPE_NEW = 48

class LabelEncoder():
    def __init__(self,):
        self._classes = []

    def fit(self,labels):
        self._classes = list(set(labels))

    def transform(self,labels):
        return np.fromiter(map(self._classes.index,labels),dtype=np.int)

def save_score(y,y_pred):
    scores = dict(
        accuracy = accuracy_score(y,y_pred),
        precision = precision_score(y,y_pred),
        recall = recall_score(y,y_pred),
    )
    print (scores)
    json.dump(open("./score.json","w+"),scores)



def crop(img,pad=True):
    W_THRESH = 8
    H_THRESH = 8
    PAD = 3 if pad else 0

    W_MIN,W_MAX = np.where(img.std(axis=0) > W_THRESH)[0][[0,-1]]
    H_MIN,H_MAX = np.where(img.std(axis=1) > H_THRESH)[0][[0,-1]]
    
    return np.pad(img[H_MIN:H_MAX,W_MIN:W_MAX],PAD,constant_values=253)

def resize(img,shape_=SHAPE_NEW,crop_=True,inv_=True):
    
    if crop_:
        img = crop(img.reshape(SHAPE).astype(np.uint8))
    if inv_:
        ret,img = cv.threshold(img,110,255,cv.THRESH_BINARY_INV)
        
    return cv.resize(img,(shape_,shape_)).astype(np.uint8)

def input_flow(X,sharpen=1):
    for i in range(X.shape[0]):
        row = X.iloc[i].values
        yield ({
                'input':resize(row[1:-4]).reshape(1,SHAPE_NEW,SHAPE_NEW,1)/255
            },
            {
                'grapheme_root':grapheme_root_ohe.transform([row[-4:-3]]),
                'vowel_diacritic':vowel_diacritic_ohe.transform([row[-3:-2]]),
                'consonant_diacritic':consonant_diacritic_ohe.transform([row[-2:-1]])
            }
        )

inputs = Input(shape = (SHAPE_NEW, SHAPE_NEW, 1),name="input")
model = Conv2D(filters=32, kernel_size=(4, 4), padding='SAME', activation='relu', input_shape=(SHAPE_NEW, SHAPE_NEW, 1))(inputs)
model = Conv2D(filters=32, kernel_size=(4, 4), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(4, 4), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(4, 4), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=256, kernel_size=(6, 6), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Flatten()(model)
model = Dense(1024, activation = "relu")(model)
model = Dropout(rate=0.3)(model)
dense = Dense(512, activation = "relu")(model)

head_root = Dense(168, activation = 'softmax',name="grapheme_root")(dense)
head_vowel = Dense(11, activation = 'softmax',name='vowel_diacritic')(dense)
head_consonant = Dense(7, activation = 'softmax',name='consonant_diacritic')(dense)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])



labels = pd.read_csv("./train.csv")

grapheme_root_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)
vowel_diacritic_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)
consonant_diacritic_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)

grapheme_root_ohe.fit(labels[['grapheme_root']])
vowel_diacritic_ohe.fit(labels[['vowel_diacritic']])
consonant_diacritic_ohe.fit(labels[['consonant_diacritic']])

EPOCHS=10

for epoch in range(EPOCHS):
    for file_id in range(4):
        df = pd.read_parquet("./train_image_data_0.parquet")
        df = pd.merge(df,labels,on='image_id')
        model.fit_generator(input_flow(df),steps_per_epoch=df.shape[0],)


notify.success()

for i in globals():
    del globals()[i]

exit()
