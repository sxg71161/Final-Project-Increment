import numpy as np
import os
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X = []
Y = []

dataset = 'dataset/no'
for root, dirs, directory in os.walk(dataset):
        for j in range(len(directory)):
            print(dataset+"/"+directory[j])
            img = cv2.imread(dataset+"/"+directory[j])
            img = cv2.medianBlur(img, 5)
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X.append(im2arr)
            Y.append(0)
            
dataset = 'dataset/yes'
for root, dirs, directory in os.walk(dataset):
        for j in range(len(directory)):
            print(dataset+"/"+directory[j])
            img = cv2.imread(dataset+"/"+directory[j])
            img = cv2.medianBlur(img, 5)
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X.append(im2arr)
            Y.append(1)
                    
X = np.asarray(X)
Y = np.asarray(Y)
np.save("data/amf.txt",X)
#np.save("data/labels.txt",Y)


np.save("model/myimg_data.txt",X)
np.save("model/myimg_label.txt",Y)


def get_feature_layer(model, data):
    total_layers = len(model.layers)    
    fl_index = total_layers-1    
    feature_layer_model = Model(inputs=model.input,outputs=model.get_layer(index=fl_index).output)    
    feature_layer_output = feature_layer_model.predict(data)
    return feature_layer_output

X = np.load("data/amf.txt.npy")
Y = np.load("data/labels.txt.npy")
Y1 = np.load("data/labels.txt.npy")

print(X.shape)
Y = to_categorical(Y)
print(Y.shape)
img = X[20].reshape(64,64,3)
cv2.imshow('ff',cv2.resize(img,(250,250)))
cv2.waitKey(0)
print("shape == "+str(X.shape))
print("shape == "+str(Y.shape))
print(Y)
X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y1 = Y1[indices]

classifier = Sequential() #alexnet transfer learning code here
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 2, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(classifier.summary())
hist = classifier.fit(X, Y, batch_size=24, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
classifier.save_weights('model/amf_weights.h5')            
model_json = classifier.to_json()
with open("model/amf.json", "w") as json_file:
    json_file.write(model_json)
f = open('model/amf.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
print(Y.ravel())
cnn_data = get_feature_layer(classifier,X)
print(cnn_data.shape)
gb = GradientBoostingClassifier()
gb = gb.fit(cnn_data, Y1)
prediction = gb.predict(cnn_data);
acc =  accuracy_score(prediction,Y1)
print(acc)




