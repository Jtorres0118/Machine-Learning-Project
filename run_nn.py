"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Joselyne, Joel, and Chandini
Date: 4/4/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import util as util
from tensorflow.python.keras import backend as K
from cnn import CNNModel
from keras import layers 
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

##################

def training_curve(training_accuracy,validation_accuracy,model_type,title,filename):
    '''
    training_accuracy : a dictionary of training accuracy seperated by model type
    validation_accuracy : a dictionary of training accuracy seperated by model type
    model_type : a string list of model name
    title : the title of the plot
    filename : A string name for the file saves and displays the plot to the user when called. 
    '''
    cnn_model = model_type
    plt.plot(training_accuracy, label = "Training accuracy for %s" % cnn_model)

    plt.plot(validation_accuracy, label = "Validation accuracy for %s" % cnn_model)

    plt.xlim(0,60)
    plt.ylim(0.0,1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig("figs/%s.pdf" %filename,format='pdf')
    plt.show()
    plt.clf()
    pass

def confusion_matrix_display(y_prob,y_test,title,filename,classes):
    '''
    This function display the confusion matrix in a pdf file
    y_prob: The predicted value from a classifer model
    title: The title of the display
    filename: The name of the filename as it is being saved
    return: none just displays and save figures
    '''
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=8) 
    matrix = confusion_matrix(y_prob,y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=classes)
    disp.plot(include_values=False)
    plt.xticks(rotation=45)
    plt.yticks(rotation = 45)
    plt.title("Confusion Matrix for %s" %title)
    plt.savefig("figs/%s.pdf" %filename,format='pdf')
    plt.show()

def createmodel(n_char,w,h):
    '''
    Creates a cnn model for image input, 
    compiles the model with categorical cross-entropy loss and the Adam optimizer
    '''
    img = layers.Input(shape= (h,w,1)) # Get image as an input of size 50,200,1
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img) #50*200
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 25*100
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 13*50
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3) #to improve the stability of model
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 7*25
    
    flat = layers.Flatten()(mp3) #convert the layer into 1-D

    outs = []
    for i in range(5): #for 5 letters of captcha
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1) #drops 0.5 fraction of nodes
        res = layers.Dense(n_char, activation='sigmoid')(drop)

        outs.append(res) #result of layers
    
    # Compile model and return it
    model = Model(img, outs) #create model
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model

def agent_p(y_label,model,encode_dict,num_dict,data_name):
    '''Make predictions based off true labels'''
    pred_lst = []
    encode_label = []
    flat_label = []
    flat_bread =[]#flattened predictions
    for label in y_label:
        encode_label.append([encode_dict[l] for l in label ])
        path = 'Data/%s/%s.png'%(data_name,label)
        pred_lst.append(util.predict_image(path,model,num_dict))
    
    for i,label in enumerate(encode_label):
        flat_label.extend(label)
        flat_bread.extend(pred_lst[i])
    return flat_bread,flat_label
    

def main():
    # Invoke the above function to get our data.
    opts = util.parse_args()
    path = opts.filename

    if path == 'Data/capcha_reg/*.png' :
        data_name = 'capcha_reg'
    else: data_name = 'capcha_noise'
    X,y,n_char,encode_dict,y_true,w,h =  util.preprocessing(path)
    classes = encode_dict.keys()

    eighty_pct = int(len(X)*.8)
    X_train, y_train = X[:eighty_pct], y[:, :eighty_pct]
    indices = np.arange(X_train.shape[0])#shuffles training data
    indice = np.arange(y_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[:,indices]
    X_test, y_test = X[eighty_pct:], y[:, eighty_pct:]# initializes test data
    test_label = y_true[eighty_pct:]

    # model=createmodel(n_char,w,h)
    # Convert data types to float32
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Create an instance of CNNmodel
    cnn_model = CNNModel.create_model(n_char, w, h)
    #cnn_model = CNNmodel(n_char, w, h)
    # Train the model
    history = cnn_model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=60, validation_split=0.2)
    training_accuracy = history.history['dense_1_accuracy']
    validation_accuracy = history.history['val_dense_1_accuracy']
    curve_title = '%s Training Curve for 60 Epoch'% data_name
    training_curve(training_accuracy,validation_accuracy,'cnn',curve_title,'Training Curve for %s'%data_name)
    
    # # Evaluate the model
    # preds = cnn_model.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

    # # Perform predictions
    # cnn_preds = np.array(cnn_model.predict(X_test))
    # pred,t_label = agent_p(test_label,cnn_model,encode_dict,len(classes),data_name)

    # #confusion_matrix_display(pred,t_label,"CNN Model", "Matrix CNN Model",classes)
    # if data_name == 'capcha_reg':
    #     title = "Regular Captcha CNN Model"
    # else: title = "Noise Captcha CNN Model"
    # confusion_matrix_display(pred,t_label,title, title,classes)

main()