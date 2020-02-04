#!/usr/bin/env python
# coding: utf-8

# In[23]:


from warnings import filterwarnings
filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm_notebook
from notifyme import notify

import gc


# In[3]:


labels = pd.read_csv("./train.csv")


# In[4]:


labels.nunique()


# In[5]:


grapheme_root_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)
vowel_diacritic_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)
consonant_diacritic_ohe = OneHotEncoder(dtype=np.uint16,sparse=False)

grapheme_root_ohe.fit(labels[['grapheme_root']])
vowel_diacritic_ohe.fit(labels[['vowel_diacritic']])
consonant_diacritic_ohe.fit(labels[['consonant_diacritic']])


# In[7]:


inputs = Input(shape = (64, 64, 1),name="inputs")
model = Conv2D(filters=32, kernel_size=(4, 4), padding='SAME', activation='relu', input_shape=(64, 64, 1))(inputs)
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


# In[8]:


model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])


# In[9]:


def crop(img,pad=True):
    W_THRESH = 8
    H_THRESH = 8
    PAD = 3 if pad else 0

    W_MIN,W_MAX = np.where(img.std(axis=0) > W_THRESH)[0][[0,-1]]
    H_MIN,H_MAX = np.where(img.std(axis=1) > H_THRESH)[0][[0,-1]]
    
    return np.pad(img[H_MIN:H_MAX,W_MIN:W_MAX],PAD,constant_values=253)

def resize(img):
    img = crop(img.reshape(137,236).astype(np.uint8))
    ret,img = cv.threshold(img,110,255,cv.THRESH_BINARY_INV)    
    return cv.resize(img,(64,64)).astype(np.uint8).reshape(64,64,1)

def input_flow(x,y,batch_size=200):
    for i in range(batch_size,x.shape[0],batch_size):
        rows = x.iloc[i-batch_size:i].values
        yield (
                {"inputs":np.apply_along_axis(resize,axis=1,arr=rows)},
                {
                    "grapheme_root":y[0][i-batch_size:i],
                    'vowel_diacritic':y[1][i-batch_size:i],
                    'consonant_diacritic':y[2][i-batch_size:i]
                }
            )


# In[21]:


def get_train_test(file_id):
    df = pd.merge(
            pd.read_parquet(f"./train_image_data_{file_id}.parquet"),
            labels,
            on='image_id'
        )
    
    grapheme_root = grapheme_root_ohe.transform(df.grapheme_root.reshape(-1,1))
    vowel_diacritic = vowel_diacritic_ohe.transform(df.vowel_diacritic.reshape(-1,1))
    consonant_diacritic = consonant_diacritic_ohe.transform(df.consonant_diacritic.reshape(-1,1))
    
    df = df.drop(columns=['image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme'])
    
    return df,(grapheme_root,vowel_diacritic,consonant_diacritic)


# In[28]:


X,Y = get_train_test(0)


# In[31]:


BATCH_SIZE = 500
EPOCHS = 10

gen = input_flow(X,Y,batch_size=BATCH_SIZE)

model.fit_generator(gen,steps_per_epoch=X.shape[0]//BATCH_SIZE)


# In[ ]:




