from PIL import Image, ImageOps
from numpy import asarray
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import optparse
import sys
import tensorflow as tf
import os

"""
COMMAND LINE ARGUMENT:  
python util.py -r "Data/capcha_reg/*.png  
OR    
python3 util.py -r "Data/capcha_noise/*.png""
"""

    
def set_characters(data):
    '''
    Takes a string of text and creates a dictionary of chars and symbols 
    data: a string of text
    return: an array of dictionary of chars 
    '''
    char_dict = []
    for file in data:
        for char in file:
            if len(char_dict) == 0 :
                char_dict.append(char)
            else:
                if char not in char_dict:
                    char_dict.append(char)
    char_dict.sort()

    return char_dict,len(char_dict)

def mapping_dict(char_dict):
    '''
    Assign unique integers on an existing dictionary
    char_dict : An array of a char dictionary
    return : Two dictionary of encoding and decoding styles
    '''
    enc_dict = dict()
    dec_dict = dict()
    for key, char in enumerate(char_dict):
        enc_dict[char] = key
        dec_dict[key] = char

    return enc_dict, dec_dict

def encode(data,dict):
    '''encodes a string into integers'''
    int_text =[]
    for file in data:
        arr_text = []
        str_text = ""
        for char in file:
            arr_text.append(dict[char])
            str_text += str(dict[char]) + " "
        int_text.append(arr_text)
    return int_text

    
def file_breakdown(file):
    '''opens the file and collects data of the filenames'''
    y_true = []
    for name in file:
        img=Image.open(name)
        y_raw = os.path.basename(name)
        w,h = img.size
        if y_raw[6:9]=="png":
            y_true.append(y_raw[0:5])
    char_dict, n_char = set_characters(y_true)
    encode_dict, decode_dict = mapping_dict(char_dict)
    encode_true = encode(y_true,encode_dict)
    return encode_true,n_char,encode_dict,decode_dict,w,h

def parse_args():
    '''Parse command line arguments (train and test arff files).'''
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--filename', type='string', help='path to' +\
        'file')

    (opts, args) = parser.parse_args()

    mandatories = ['filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()
    return opts

def preprocessing(path):
    '''
    Preprocesses images from a specified path by normalizing the pixel values, 
    encoding the filenames into target arrays, and splitting the data into training, validation, and test sets
    '''
    # n : the length of files in the data 
    # nchar: number of char in the dict 
    y_true=[]
    img_list_raw= []
    file_list= glob.glob(path)
    file_list.sort()

    ###read the file and colllect its dict####
    encode_true,n_char,encode_dict,decode_dict,w,h= file_breakdown(file_list)

    X = np.zeros((len(file_list),h,w,1))
    y = np.zeros((5,len(file_list),n_char))
    for i,file in enumerate(file_list):
        img=Image.open(file)
        y_raw = os.path.basename(file)
        if y_raw[6:9]=="png":
            pic_name = y_raw[0:5]
            y_true.append(y_raw[0:5])
        img= ImageOps.grayscale(img) #read into grey scale
       
        img = asarray(img) / 255.0
        img = np.reshape(img, (h, w, 1))
        target=np.zeros((5,n_char))

        img_list_raw.append(img)
        for num,char in enumerate(pic_name):
            index = encode_dict[char]
            target[num, index] = 1
       
        X[i] = img
        y[:,i] = target
    return X,y,n_char,encode_dict,y_true,w,h

def predict_image(filepath,model,num_dic):
    '''This method predicts the label of an image'''
    img=Image.open(filepath)
    img= ImageOps.grayscale(img) #grey scale

    img = asarray(img) / 255.0 

    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis])) 
    result = np.reshape(res, (5, num_dic)) #reshape the array
    label = []
    for i in result:
        label.append(np.argmax(i))
    return label