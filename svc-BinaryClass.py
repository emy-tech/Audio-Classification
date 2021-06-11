"""
Course : ENGI 981A
Project : Classification Of Marine Mammal Sound Using Machine Learning
Written by : Chukwuemeka Achilefu
Mun ID: 201794630
Purpose: SVM Binary Classification
Date Last Modified : 24/04/2021
Software : Jupyter Notebook
"""

import pandas as pd
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
from python_speech_features import mfcc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Sequential
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Fuction to cary out MFCC extraction
def extract_features1(file_name): 
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled

#Loading of DataSet
metadata = pd.read_csv('New_sound/NewSoundBio.csv',
                      dtype={'ClassID': np.int32, 'Class_name': str})
metadata.head()

#extraction of audio features from dataset
audio_data = list()
for i in tqdm(range(metadata.shape[0])):
    audio_data.append(librosa.load(metadata['SoundPath'].iloc[i]))
audio_data = np.array(audio_data, dtype="object")

#DataSet Shaping
metadata['audio_waves'] = audio_data[:,0]
metadata['samplerate'] = audio_data[:,1]
metadata.head()

#removal of audio sounds less tahn 1 second from dataset
bit_lengths = list()
for i in range(metadata.shape[0]):
    bit_lengths.append(len(metadata['audio_waves'].iloc[i]))
bit_lengths = np.array(bit_lengths)
metadata['bit_lengths'] = bit_lengths
metadata['seconds_length'] = metadata['bit_lengths']/metadata['samplerate']
metadata.head()
metadata = metadata[metadata['seconds_length'] >= 1.0]

# Iterate through each sound file and extract the features 
features = []
for index, row in metadata.iterrows():
    file_name = row["SoundPath"]
    class_label = row["Class_name"]
    data = extract_features1(file_name)
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf1 = pd.DataFrame(features, columns=['feature','class_label'])
print('Finished feature extraction from ', len(featuresdf1), ' files')

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf1.feature.tolist())
y = np.array(featuresdf1.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

#Spliting of Dataset into Training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.3, random_state = 1)

#SVM Algorithm
classifier = SVC()
classifier.fit(x_train, y_train.argmax(axis=1))

#Prediction carried out on the test set and confusion Matrix development
y_pred = classifier.predict(x_test)
cm1 = confusion_matrix(y_test.argmax(axis=1), y_pred)

#Display of confusion Matrix
sns.heatmap(cm1, fmt='d', annot=True)

#Printing out of classification report
target_names = ['Atlantic spotted dolphin', 'Long Finned Pilot Whale'  ]
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=target_names))

#Definition if parameters for GridSearchCV
param = {'C':[1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,1000,10000],
         'class_weight':['balanced'],
        'decision_function_shape':['ovo', 'ovr'],
        'gamma':[1,0.1,0.001,0.0001]}

#KFold split definition
KF = KFold(n_splits=10)
KF=KF.get_n_splits(x_train)

#GridSearchCV
grid = GridSearchCV(estimator=classifier, param_grid=param, cv=KF, scoring="accuracy")
best_auprc_grid = grid.fit(x_train, y_train)

#finding the best score from the GridSearchCV
best_auprc_grid.best_score_

#finding the estimator from the GridSearchCV
best_auprc_grid.best_estimator_