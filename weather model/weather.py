import os
import csv
import numpy as np

import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization
from keras.layers import Flatten, Dense, MaxPooling2D
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

trainPath = 'data/TrainData-C2/'
testPath = 'data/TestData/'
training = os.listdir(trainPath)
testing = os.listdir(testPath)
trainWeather = open('ExtraCredit_Train.csv')
testWeather = open('ExtraCredit_Test.csv')

trwreader = csv.reader(trainWeather)
tereader = csv.reader(testWeather)

train_img = []
train_weather = []
test_img = []
test_weather = []
train_out = []
test_out = []
extras = training.copy() #remove the test images from the training
test_names = [] #names of the test images
maxis = [0]*12 #stores all the maximum values for normalizing inputs
all_vars = [[],[],[],[],[],[],[],[],[],[],[],[]] #separates all the data into their own list of normalization
image_size = 224 

def findMax(l, i):
    maxis[i] = max(l)

#since the weather data contains a variety of data, normalize it
def normalize(l, n):    
    for v in l:
        for i in range(12):
            value = float(v[i])
            norma = n[i]
            v[i] = (value / norma)
    
#load all the weather data
def loadDataWeather(combine):
    labels = True #checking if I'm on the first row
    for z in tereader:
        if labels:
            labels = False
            continue
        arr = int(z[1])
        var = []
        for v in range(2, len(z)):
            var.append(z[v])
        extras.remove(z[0])
        var = np.array(var)
        var.reshape(12, 1)
        img = load_img(trainPath + z[0], target_size=(image_size, image_size))            
        img = img_to_array(img)/255.0
        test_img.append(img)
        test_out.append(arr)
        test_weather.append(var)
        
    all_vars = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for l in test_weather:
        for i in range(12):
            all_vars[i].append(float(l[i]))
    for i in range(12):
        findMax(all_vars[i], i) #find the largest value in the list
        
    normalize(test_weather, maxis) #normalize all the data
    
    labels = True
    for y in trwreader:
        if labels:
            labels = False
            continue
            
        arr = int(y[1])
        var = []
        for v in range(2, len(y)):
            var.append(y[v])
        extras.remove(y[0])
        var = np.array(var)
        var.reshape(-1, 1)
        img = load_img(trainPath + y[0], target_size=(image_size, image_size))            
        img = img_to_array(img)/255.0
        train_img.append(img)
        train_out.append(arr)
        train_weather.append(var)
    
    #reset all variables for the training data
    all_vars = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for l in train_weather:
        for i in range(12):
            all_vars[i].append(float(l[i]))
    for i in range(12):
        findMax(all_vars[i], i)
        
    normalize(train_weather, maxis)       
        
    return (train_weather, test_weather), (train_img, train_out), (test_img, test_out)
    
#weather data MLP
model = Sequential()
#using input_dim=12 because there are 12 data points per image
model.add(Dense(1024, activation='relu', input_dim=12))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

#image classifier CNN
cnn = Sequential()
cnn.add(Conv2D(32, [3,3], padding='same', 
                      input_shape=(image_size, image_size, 3)))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, [3,3], strides=(2,2), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, [3,3], strides=(2,2), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, [3,3], strides=(2,2), activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(Flatten())
cnn.add(Dense(5, activation='softmax'))
cnn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0003),
                  metrics=['accuracy'])

#combining the two
out = keras.layers.concatenate([cnn.output, model.output])
out = Dense(10, activation='relu')(out)
out = Dense(5, activation='softmax')(out)
final_model = Model(inputs=[model.input, cnn.input], outputs=out)
final_model.compile(loss='categorical_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])
final_model.summary()

epochs = 15

(w_train, w_test), (x_train, y_train), (x_test, y_test) = loadDataWeather(True)

#reshape to fit model
x_train = np.array(x_train).reshape(len(x_train), image_size, image_size, 3)
w_train = np.array(w_train).reshape(-1, 12)
w_test = np.array(w_test).reshape(-1, 12)
x_test = np.array(x_test).reshape(len(x_test), image_size, image_size, 3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# #end reshape


combine_history = final_model.fit([w_train, x_train], 
                          y_train,
                          batch_size=32,
                          validation_split=0.1,
                          epochs=epochs,
                          verbose=1)

#get final accuracy and loss of the model
pred_train= final_model.predict([w_train, x_train])
scores = final_model.evaluate([w_train, x_train], y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1] * 100, 1 - scores[1]))   
 
pred_test= final_model.predict([w_test, x_test])
scores2 = final_model.evaluate([w_test, x_test], y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1] * 100, 1 - scores2[1]))


#plot the loss and accuracy graphs
plot_range = list(range(len(combine_history.history['loss'])))
plt.figure()
plt.plot(plot_range, combine_history.history['loss'])
plt.plot(plot_range, combine_history.history['val_loss'])
plt.legend(('Combined Train Loss', 'Combined Val Loss'))
plt.show()

plt.figure()
plt.plot(plot_range, combine_history.history['accuracy'])
plt.plot(plot_range, combine_history.history['val_accuracy'])
plt.legend(('Combined Train Accuracy', 'Combined Val Accuracy'))
plt.show()            
#end plotting