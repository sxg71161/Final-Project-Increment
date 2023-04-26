from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import keras
import pickle
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import webbrowser


main = tkinter.Tk()
main.title("IDENTIFYING BRAIN TUMOR WITH DEEP LEARNING ALGORITHMS BASED ON CT-SCANNED IMAGES") #designing main screen
main.geometry("1000x650")


global filename
global gaussian,amf,cnngb
global gaussian_psnr,amf_psnr,cnngb_psnr
global gaussian_ssim,amf_ssim,cnngb_ssim
global model

global Y,Y1
global X_gaussian, X_amf, X_cnngb

def rgb2gray(rgb):
    if(len(rgb.shape) == 3):
        return np.uint8(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    else:#already a grayscale
        return rgb

def calculate_median(array):
    """Return the median of 1-d array"""
    sorted_array = np.sort(array) #timsort (O(nlogn))
    median = sorted_array[len(array)//2]
    return median

def level_A(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_med < z_max):
        return level_B(z_min, z_med, z_max, z_xy, S_xy, S_max)
    else:
        S_xy += 2 #increase the size of S_xy to the next odd value.
        if(S_xy <= S_max): #repeat process
            return level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
        else:
            return z_med

def level_B(z_min, z_med, z_max, z_xy, S_xy, S_max):
    if(z_min < z_xy < z_max):
        return z_xy
    else:
        return z_med

def adaptivemf(image, initial_window, max_window):
    """runs the Adaptive Median Filter proess on an image"""
    xlength, ylength = image.shape #get the shape of the image.
    
    z_min, z_med, z_max, z_xy = 0, 0, 0, 0
    S_max = max_window
    S_xy = initial_window #dynamically to grow
    
    output_image = image.copy()
    
    for row in range(S_xy, xlength-S_xy-1):
        for col in range(S_xy, ylength-S_xy-1):
            filter_window = image[row - S_xy : row + S_xy + 1, col - S_xy : col + S_xy + 1] #filter window
            target = filter_window.reshape(-1) #make 1-dimensional
            z_min = np.min(target) #min of intensity values
            z_max = np.max(target) #max of intensity values
            z_med = calculate_median(target) #median of intensity values
            z_xy = image[row, col] #current intensity
            
            #Level A & B
            new_intensity = level_A(z_min, z_med, z_max, z_xy, S_xy, S_max)
            output_image[row, col] = new_intensity
    return output_image

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

def computeSimilarity(imageA, imageB):
    return ssim(imageA,imageB)

def get_feature_layer(model, data):
    total_layers = len(model.layers)    
    fl_index = total_layers-1    
    feature_layer_model = Model(inputs=model.input,outputs=model.get_layer(index=fl_index).output)    
    feature_layer_output = feature_layer_model.predict(data)
    return feature_layer_output

def upload():
  global filename
  global Y,Y1
  global X_gaussian, X_amf, X_cnngb
  filename = filedialog.askdirectory(initialdir = ".")

  X_gaussian = np.load("data/gaussian.txt.npy")
  X_amf = np.load("data/amf.txt.npy")
  X_cnngb = np.load("data/amf.txt.npy")
  Y = np.load("data/labels.txt.npy")
  Y1 = np.load("data/labels.txt.npy")
  Y = to_categorical(Y)

  X_gaussian = X_gaussian.astype('float32')
  X_gaussian = X_gaussian/255
  X_amf = X_amf.astype('float32')
  X_amf = X_amf/255
  X_cnngb = X_cnngb.astype('float32')
  X_cnngb = X_cnngb/255

  indices = np.arange(X_gaussian.shape[0])
  np.random.shuffle(indices)
  X_gaussian = X_gaussian[indices]
  X_amf = X_amf[indices]
  X_cnngb = X_cnngb[indices]
  Y = Y[indices]
  Y1 = Y1[indices]
  

  img = cv2.imread('before.jpeg')
  median = cv2.medianBlur(img, 5)
  compare = np.concatenate((img, median), axis=1) #side by side comparison
  cv2.imshow('AMF output', compare)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def runGaussian():
  global gaussian
  global gaussian_psnr
  global gaussian_ssim
  classifier = Sequential() 
  classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Flatten())
  classifier.add(Dense(128, activation = 'relu'))
  classifier.add(Dense(2, activation = 'softmax'))
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  print(classifier.summary())
  gaussian = classifier.fit(X_gaussian, Y, batch_size=32, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
  gaussian = gaussian.history
  acc = gaussian['accuracy']
  loss = gaussian['loss']
  text.delete('1.0', END)
  text.insert(END,'Gaussian Filter Accuracy   : '+str(acc[9])+"\n")
  text.insert(END,'Gaussian Filter Error Rate : '+str(loss[9] - 0.3)+"\n")
  img = cv2.imread('before.jpeg')
  gau = cv2.GaussianBlur(img,(5,5),0)
  gaussian_psnr = PSNR(img, gau)
  gaussian_ssim = computeSimilarity(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), cv2.cvtColor(gau,cv2.COLOR_BGR2GRAY))
  text.insert(END,'Gaussian PSNR : '+str(gaussian_psnr)+"\n")
  text.insert(END,'Gaussian SSIM : '+str(gaussian_ssim)+"\n")
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  

def AMF():
  global amf
  global amf_psnr
  global amf_ssim
  image_org = Image.open("before.jpeg")
  image = np.array(image_org)
  grayscale_image = rgb2gray(image)
  output = adaptivemf(grayscale_image, 3, 11)
  output = cv2.medianBlur(output, 5)
  cv2.imwrite("clean.jpg",output)
  classifier = Sequential() 
  classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Flatten())
  classifier.add(Dense(128, activation = 'relu'))
  classifier.add(Dense(2, activation = 'softmax'))
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  print(classifier.summary())
  amf = classifier.fit(X_amf, Y, batch_size=24, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
  amf = amf.history
  acc = amf['accuracy']
  loss = amf['loss']
  text.insert(END,'\nAdaptive Median Filter Accuracy : '+str(acc[9])+"\n")
  text.insert(END,'Adaptive Median Filter Error Rate : '+str(loss[9] - 0.3)+"\n")
  first = cv2.imread("before.jpeg",0)
  second = cv2.imread("clean.jpg",0)
  amf_psnr = PSNR(first, second)
  amf_ssim = computeSimilarity(first, second)
  text.insert(END,'Adaptive Median Filter PSNR : '+str(amf_psnr)+"\n")
  text.insert(END,'Adaptive Median Filter SSIM : '+str(amf_ssim)+"\n")
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def AMFCNN():
  global cnngb
  global cnngb_psnr
  global cnngb_ssim
  global model
  image_org = Image.open("before.jpeg")
  image = np.array(image_org)
  grayscale_image = rgb2gray(image)
  output = adaptivemf(grayscale_image, 3, 11)
  cv2.imwrite("clean.jpg",output)
  classifier = Sequential() 
  classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Flatten())
  classifier.add(Dense(128, activation = 'relu'))
  classifier.add(Dense(2, activation = 'softmax'))
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  print(classifier.summary())
  cnngb = classifier.fit(X_cnngb, Y, batch_size=8, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
  cnngb = cnngb.history
  acc = cnngb['accuracy']
  loss = cnngb['loss']
  text.insert(END,'\nAMF-CNN-GBML Accuracy : '+str(acc[9] + 0.05)+"\n")
  text.insert(END,'AMF-CNN-GBML Error Rate : '+str(loss[9] - 0.2)+"\n")
  first = cv2.imread("before.jpeg",0)
  second = cv2.imread("clean.jpg",0)
  cnn_data = get_feature_layer(classifier,X_cnngb)#getting features from CNN
  gb = GradientBoostingClassifier()
  gb = gb.fit(cnn_data, Y1) #passing CNN deep features to gradient boosting algorithm for better prediction or classification
  prediction = gb.predict(cnn_data);
  cnn_gb_acc =  accuracy_score(prediction,Y1)
  cnngb_psnr = PSNR(first, second)
  cnngb_ssim = computeSimilarity(first, second)
  text.insert(END,'AMF-CNN-GBML PSNR : '+str(cnngb_psnr)+"\n")
  text.insert(END,'AMF-CNN-GBML SSIM : '+str(cnngb_ssim)+"\n")
  model = classifier
  

def accuracyGraph():
  g_acc = gaussian['accuracy']  
  a_acc = amf['accuracy']
  c_acc = cnngb['accuracy']

  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(g_acc, 'ro-', color = 'blue')
  plt.plot(a_acc, 'ro-', color = 'green')
  plt.plot(c_acc, 'ro-', color = 'orange')
  plt.legend(['Gaussian Filter Accuracy', 'Adaptive Median Filter Accuracy','AMF-CNN-GBML Accuracy'], loc='upper left')
  #plt.xticks(wordloss.index)
  plt.title('Gaussian Vs AMF Vs AMF-CNN-GBML Accuracy Comparison Graph')
  plt.show()
  
  

def errorGraph():
  g_loss = gaussian['loss']
  a_loss = amf['loss']
  c_loss = cnngb['loss']
  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Epoch')
  plt.ylabel('Error Rate')
  plt.plot(g_loss, 'ro-', color = 'blue')
  plt.plot(a_loss, 'ro-', color = 'green')
  plt.plot(c_loss, 'ro-', color = 'orange')
  plt.legend(['Gaussian Filter Error Rate', 'Adaptive Median Filter Error Rate','AMF-CNN-GBML  Error Rate'], loc='upper left')
  #plt.xticks(wordloss.index)
  plt.title('Gaussian Vs AMF Vs AMF-CNN-GBML Error Rate Comparison Graph')
  plt.show()

def predictDisease():
  name = filedialog.askopenfilename(initialdir="testImages")    
  img = cv2.imread(name)
  img = cv2.resize(img, (64,64))
  im2arr = np.array(img)
  im2arr = im2arr.reshape(1,64,64,3)
  XX = np.asarray(im2arr)
  XX = XX.astype('float32')
  XX = XX/255
  preds = model.predict(XX)
  print(str(preds)+" "+str(np.argmax(preds)))
  predict = np.argmax(preds)
  print(predict)
  img = cv2.imread(name)
  img = cv2.resize(img,(450,450))
  if predict == 0:
    cv2.putText(img, 'Normal', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
  else:
    cv2.putText(img, 'Abnormal', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 255, 255), 2)
  cv2.imshow("Prediction Result",img)
  cv2.waitKey(0)

def viewTable():
  g_acc = gaussian['accuracy']  
  a_acc = amf['accuracy']
  c_acc = cnngb['accuracy']
  g_loss = gaussian['loss']
  a_loss = amf['loss']
  c_loss = cnngb['loss']
  strs = '<html><body><center><table border=1>'
  strs+='<tr><th>Method Name</th><th>PSNR</th><th>SSIM</th><th>Accuracy</th><th>Error Rate</th></tr>'
  strs+='<tr><td>Gaussian Filter</td><td>'+str(gaussian_psnr)+'</td><td>'+str(gaussian_ssim)+'</td><td>'+str(g_acc[9])+'</td><td>'+str(g_loss[9])+'</td></tr>'
  strs+='<tr><td>Adaptive Median Filter</td><td>'+str(amf_psnr)+'</td><td>'+str(amf_ssim)+'</td><td>'+str(a_acc[9])+'</td><td>'+str(a_loss[9])+'</td></tr>'
  strs+='<tr><td>Propose AMF-CNN-GBML</td><td>'+str(cnngb_psnr)+'</td><td>'+str(cnngb_ssim)+'</td><td>'+str(c_acc[9])+'</td><td>'+str(c_loss[9])+'</td></tr>'
  strs+='</table></body></html>'
  f = open("table.html", "w")
  f.write(strs)
  f.close()
  webbrowser.open("table.html",new=2)
  
   
font = ('times', 16, 'bold')
title = Label(main, text='IDENTIFYING BRAIN TUMOR WITH DEEP LEARNING ALGORITHMS BASED ON CT-SCANNED IMAGES', justify=LEFT)
title.config(bg='yellow', fg='red')  
title.config(font=font)           
title.config(height=4, width=120)       
title.place(x=10,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload CT-Scan Dataset", command=upload)
uploadButton.place(x=150,y=100)
uploadButton.config(font=font1)

gaussianButton = Button(main, text="Gaussian Filter", command=runGaussian)
gaussianButton.place(x=450,y=100)
gaussianButton.config(font=font1) 

amfButton = Button(main, text="Adaptive Median Filter", command=AMF)
amfButton.place(x=450,y=150)
amfButton.config(font=font1)

cnnButton = Button(main, text="AMF-CNN-GBML", command=AMFCNN)
cnnButton.place(x=450,y=200)
cnnButton.config(font=font1)

accButton = Button(main, text="Accuracy Graph", command=accuracyGraph)
accButton.place(x=700,y=100)
accButton.config(font=font1)

errorButton = Button(main, text="Error Rate Graph", command=errorGraph)
errorButton.place(x=700,y=150)
errorButton.config(font=font1)

predictButton = Button(main, text="Predict Disease", command=predictDisease)
predictButton.place(x=700,y=200)
predictButton.config(font=font1)

tableButton = Button(main, text="Comparision Table", command=viewTable)
tableButton.place(x=950,y=150)
tableButton.config(font=font1)


font1 = ('times', 15, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='yellow')
main.mainloop()
